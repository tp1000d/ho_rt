"""
Compatibility shim that lets Sionna 0.18 run on top of DrJit 1.x / Mitsuba 3.8,
plus a wrapper around Scene.compute_fields() so notebooks written for the
pre-RIS Sionna API (6-tuple traced_paths) still work.

Patches:
  1. dr.reinterpret_array_v(dtype, src) -> dr.reinterpret_array(dtype, src).
  2. sionna.rt.utils.mi_to_tf_tensor: transpose drjit-1.x (C, N) layout back to
     the (N, C) layout that the rest of Sionna 0.18 expects + handle Bool DLPack.
  3. Scene.compute_fields(*spec,diff,scat, *spec_tmp,diff_tmp,scat_tmp): inject
     empty RIS path objects when called with the legacy 6-arg signature.

The layout fixes for the few solver files that pass (N, C)-layout tensors into
drjit constructors are applied as direct source edits in those files.
"""
from __future__ import annotations

import drjit as dr


# --- 1. dr.reinterpret_array_v alias ----------------------------------------
if not hasattr(dr, "reinterpret_array_v") and hasattr(dr, "reinterpret_array"):
    dr.reinterpret_array_v = dr.reinterpret_array


# --- 2. mi_to_tf_tensor: ensure (N, C) output for vector-typed arrays -------
def _patch_mi_to_tf_tensor():
    from . import utils as _su
    import tensorflow as tf
    import mitsuba as mi

    def mi_to_tf_tensor(mi_tensor, dtype):
        dr.eval(mi_tensor)
        dr.sync_thread()
        src = mi_tensor
        try:
            if dr.type_v(src) == dr.VarType.Bool:
                src = dr.select(src, mi.UInt8(1), mi.UInt8(0))
        except Exception:
            pass
        shape = dr.shape(src)
        if len(shape) >= 1 and shape[-1] == 1:
            src = dr.repeat(src, 2)
            tf_tensor = tf.cast(src.tf(), dtype)[:1]
        else:
            tf_tensor = tf.cast(src.tf(), dtype)
        # Drjit 1.x reports Vector{2,3,4}f as (C, N). Transpose to legacy (N, C).
        shape = dr.shape(src)
        if len(shape) == 2 and shape[0] in (2, 3, 4) and shape[0] != shape[1]:
            if tuple(tf_tensor.shape) == tuple(shape):
                tf_tensor = tf.transpose(tf_tensor)
        return tf_tensor

    _su.mi_to_tf_tensor = mi_to_tf_tensor


_patch_mi_to_tf_tensor()


# --- 3. Scene.compute_fields legacy 6-arg signature -------------------------
# Sionna 0.18 added RIS path objects to trace_paths()/compute_fields() (now an
# 8-tuple). The diff-rt-calibration notebooks were written against the 6-tuple
# API. Wrap compute_fields so a 6-arg call works by injecting empty RIS path
# placeholders.
def _patch_compute_fields():
    from .scene import Scene
    from .paths import Paths
    from .solver_paths import PathsTmpData

    _orig = Scene.compute_fields

    def compute_fields(self, *args, **kwargs):
        if len(args) == 6:
            spec, diff, scat, spec_tmp, diff_tmp, scat_tmp = args
            sources = spec.sources
            targets = spec.targets
            ris = Paths(sources=sources, targets=targets, scene=self,
                        types=Paths.RIS)
            ris_tmp = PathsTmpData(sources, targets, self._dtype)
            args = (spec, diff, scat, ris,
                    spec_tmp, diff_tmp, scat_tmp, ris_tmp)
        return _orig(self, *args, **kwargs)

    compute_fields.__wrapped__ = _orig
    Scene.compute_fields = compute_fields


_patch_compute_fields()

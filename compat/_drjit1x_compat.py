"""
Compatibility shim that lets Sionna 0.18 run on top of DrJit 1.x / Mitsuba 3.8.

Two compat fixes only — and we explicitly do NOT replace mi.Point2f/Vector3f
at the module level, since downstream code passes those classes as the dtype
argument to dr.gather, which requires a real type, not a Python function.

Patches:
  1. dr.reinterpret_array_v(dtype, src) -> dr.reinterpret_array(dtype, src).
  2. sionna.rt.utils.mi_to_tf_tensor: transpose drjit-1.x (C, N) layout back to
     the (N, C) layout that the rest of Sionna 0.18 expects.

The remaining layout fixes are applied as targeted edits to the few solver
files where Sionna passes (N, C)-layout tensors into drjit constructors that
now require (C, N).
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

    def mi_to_tf_tensor(mi_tensor, dtype):
        dr.eval(mi_tensor)
        dr.sync_thread()
        shape = dr.shape(mi_tensor)
        if len(shape) >= 1 and shape[-1] == 1:
            mi_tensor = dr.repeat(mi_tensor, 2)
            tf_tensor = tf.cast(mi_tensor.tf(), dtype)[:1]
        else:
            tf_tensor = tf.cast(mi_tensor.tf(), dtype)
        # Drjit 1.x reports Vector{2,3,4}f as (C, N). Transpose to legacy (N, C).
        if len(shape) == 2 and shape[0] in (2, 3, 4) and shape[0] != shape[1]:
            if tuple(tf_tensor.shape) == tuple(shape):
                tf_tensor = tf.transpose(tf_tensor)
        return tf_tensor

    _su.mi_to_tf_tensor = mi_to_tf_tensor


_patch_mi_to_tf_tensor()

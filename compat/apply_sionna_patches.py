"""
Apply the Sionna 0.18 + DrJit 1.x compat patches to a freshly installed env.

Run this ONCE inside the Singularity container after `pip install sionna==0.18.0
drjit==1.3.1 mitsuba==3.8.0` so that the source files are patched in place.
The setup_overlay sbatch invokes it automatically.

Patches:
  1. Drop `_drjit1x_compat.py` next to `sionna/rt/__init__.py` and import it.
     The shim itself fixes:
       - dr.reinterpret_array_v -> dr.reinterpret_array alias.
       - mi_to_tf_tensor: (C,N) -> (N,C) layout, bool DLPack via UInt8.
       - Scene.compute_fields: accept legacy 6-arg call, inject empty RIS
         placeholders (8-arg API was added in Sionna 0.18 with RIS support).
  2. Wrap (N, C) -> (C, N) layout-sensitive constructor call sites with
     `_ls(...)` from `sionna.rt.utils`.
  3. Replace mi.Color0f(0.) with mi.Color0f().
  4. Cast `prims_i` sentinel to Int32 so `-1` doesn't fail UInt promotion.
  5. Patch `mi_to_tf_tensor` source to mirror the runtime shim (so it works
     even when the shim is bypassed).
"""
import re
import shutil
import sys
from pathlib import Path


def _site_packages():
    for p in sys.path:
        pp = Path(p)
        if pp.name == "site-packages" and (pp / "sionna").exists():
            return pp
    raise RuntimeError("Cannot locate sionna in sys.path")


def _patch_init(rt_dir: Path, this_dir: Path):
    init_py = rt_dir / "__init__.py"
    src = init_py.read_text()
    if "_drjit1x_compat" in src:
        return False
    shutil.copy(this_dir / "_drjit1x_compat.py", rt_dir / "_drjit1x_compat.py")
    init_py.write_text(
        src.rstrip()
        + "\n\n# DrJit 1.x / Mitsuba 3.8 compatibility shim for Sionna 0.18.\n"
        + "from . import _drjit1x_compat  # noqa: F401\n"
    )
    return True


def _patch_utils(rt_dir: Path):
    utils = rt_dir / "utils.py"
    src = utils.read_text()
    if "_ls(" in src:
        return False
    helper = '''
def _ls(x):
    """drjit-1.x layout swap: (N, C) -> (C, N) for C in {2, 3, 4}.

    Returns ``x`` unchanged if it does not look like a per-row vector tensor.
    """
    shape = getattr(x, "shape", None)
    if shape is None or len(shape) != 2:
        return x
    if shape[-1] in (2, 3, 4) and shape[0] != shape[-1]:
        return tf.transpose(x)
    return x


'''
    src = src.replace("def mi_to_tf_tensor(", helper + "def mi_to_tf_tensor(", 1)
    # Replace mi_to_tf_tensor body to transpose + bool->UInt8.
    src = src.replace(
        '''def mi_to_tf_tensor(mi_tensor, dtype):
    """
    Get a TensorFlow eager tensor from a Mitsuba/DrJIT tensor
    """
    dr.eval(mi_tensor)
    dr.sync_thread()
    # When there is only one input, the .tf() methods crashes.
    # The following hack takes care of this corner case
    if dr.shape(mi_tensor)[-1] == 1:
        mi_tensor = dr.repeat(mi_tensor, 2)
        tf_tensor = tf.cast(mi_tensor.tf(), dtype)[:1]
    else:
        tf_tensor = tf.cast(mi_tensor.tf(), dtype)
    return tf_tensor''',
        '''def mi_to_tf_tensor(mi_tensor, dtype):
    """drjit-1.x compatible Mitsuba->TF tensor conversion."""
    dr.eval(mi_tensor)
    dr.sync_thread()
    src = mi_tensor
    try:
        if dr.type_v(src) == dr.VarType.Bool:
            src = dr.select(src, mi.UInt8(1), mi.UInt8(0))
    except Exception:
        pass
    if dr.shape(src)[-1] == 1:
        src = dr.repeat(src, 2)
        tf_tensor = tf.cast(src.tf(), dtype)[:1]
    else:
        tf_tensor = tf.cast(src.tf(), dtype)
    shape = dr.shape(src)
    if len(shape) == 2 and shape[0] in (2, 3, 4) and shape[0] != shape[1]:
        if tuple(tf_tensor.shape) == tuple(shape):
            tf_tensor = tf.transpose(tf_tensor)
    return tf_tensor''',
    )
    utils.write_text(src)
    return True


def _patch_layout_sensitive(rt_dir: Path):
    """Wrap self._mi_(point|point2|vec)_t(EXPR) with _ls(EXPR)."""
    pat = re.compile(
        r"(self\._mi_(?:point|point2|vec)_t)\(([^()]*?(?:\([^()]*\)[^()]*?)*?)\)"
    )
    n_total = 0
    for fn in ("scene_object.py", "solver_base.py", "solver_cm.py", "solver_paths.py"):
        p = rt_dir / fn
        src = p.read_text()
        new_src, n = pat.subn(
            lambda m: f"{m.group(1)}(_ls({m.group(2)}))" if "_ls(" not in m.group(2) else m.group(0),
            src,
        )
        if n > 0:
            if "from .utils import _ls" not in new_src and "_ls" not in new_src.split("def ", 1)[0]:
                if "from .utils import" in new_src:
                    new_src = new_src.replace("from .utils import", "from .utils import _ls,", 1)
                else:
                    new_src = "from .utils import _ls\n" + new_src
            p.write_text(new_src)
            n_total += n
    return n_total


def _patch_solver_base_color0f(rt_dir: Path):
    p = rt_dir / "solver_base.py"
    src = p.read_text()
    if "mi.Color0f(0.)" in src:
        src = src.replace("mi.Color0f(0.)", "mi.Color0f()")
        p.write_text(src)
        return True
    return False


def _patch_solver_paths_int32(rt_dir: Path):
    p = rt_dir / "solver_paths.py"
    src = p.read_text()
    needle = "prims_i = dr.select(active, offsets + si.prim_index, -1)"
    repl = (
        "prims_i = dr.select(active,\n"
        "                                    offsets + mi.Int32(si.prim_index),\n"
        "                                    mi.Int32(-1))"
    )
    if needle in src:
        src = src.replace(needle, repl)
        p.write_text(src)
        return True
    return False


def main():
    site = _site_packages()
    rt = site / "sionna" / "rt"
    if not rt.exists():
        raise RuntimeError(f"sionna/rt not found at {rt}")
    here = Path(__file__).resolve().parent
    print(f"site-packages: {site}")

    print("[1] init.py:", _patch_init(rt, here))
    print("[2] utils.py / _ls / mi_to_tf_tensor:", _patch_utils(rt))
    print("[3] _ls() wraps applied:", _patch_layout_sensitive(rt))
    print("[4] Color0f():", _patch_solver_base_color0f(rt))
    print("[5] Int32 prims_i sentinel:", _patch_solver_paths_int32(rt))
    print("Done.")


if __name__ == "__main__":
    main()

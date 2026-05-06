"""
Patch the diff-rt-calibration figure notebooks so that nbconvert --execute
works end-to-end without manual intervention.

The upstream notebooks (CDFs / Heat_Maps / CIRs) have a cell that does:

    with open(result_filename, 'rb') as f:
        RT_CDFs, MS_CDFs, Err_CDFs = pickle.load(f)

right *before* the cells that initialize empty dicts and re-compute. The
upstream design assumes a human picks one of two paths manually. For
headless `nbconvert --execute --inplace`, that load cell raises
FileNotFoundError on a fresh checkout because there's no cached .pkl yet.

Fix: convert that load cell into a try/except that swallows the missing-file
error. The next cells always re-initialize the dicts and recompute, which is
what we want anyway.
"""
import json
import sys
from pathlib import Path

NB_DIR = Path("/home/wxq/Desktop/workplace/RT/ho_rt/notebooks")
TARGETS = ["CDFs.ipynb", "Heat_Maps.ipynb", "CIRs.ipynb"]


def patch(path: Path):
    nb = json.loads(path.read_text())
    changed = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        # Match the unconditional pickle load. Multiple notebooks have it.
        if "pickle.load(f)" in src and "with open(result_filename, 'rb')" in src:
            indented = "    " + src.replace("\n", "\n    ").rstrip() + "\n"
            new_src = (
                "# AUTO-PATCH: skip pickle load for fresh run; cells below recompute.\n"
                "try:\n"
                + indented
                + "    print('Loaded cached results from', result_filename)\n"
                + "except FileNotFoundError:\n"
                + "    print('No cached results at', result_filename, '— will recompute.')\n"
            )
            cell["source"] = new_src.splitlines(keepends=True)
            cell["outputs"] = []
            cell["execution_count"] = None
            changed = True
    if changed:
        path.write_text(json.dumps(nb, indent=1))
    return changed


for n in TARGETS:
    p = NB_DIR / n
    if not p.exists():
        print(f"missing {p}, skipping")
        continue
    if patch(p):
        print(f"patched {n}")
    else:
        print(f"no match in {n}")

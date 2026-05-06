# ho_rt — Hoydis-style Differentiable Ray Tracing reproduction on NYU torch HPC

Reproduce *Learning Radio Environments by Differentiable Ray Tracing*
(Hoydis et al., NVIDIA, 2023) on NYU torch HPC H200 GPUs.

This is a self-contained packaging of [NVlabs/diff-rt-calibration] plus the
patches needed to run on NVIDIA driver 570 (which ships OptiX 8 ABI), wired
into the standard NYU torch HPC apptainer + ext3-overlay workflow.

```
ho_rt/
├── slurm/
│   ├── setup_overlay.sbatch     # CPU node — stage overlay, install Sionna stack, apply patches
│   ├── download_dichasus.sbatch # CPU node — 16-stream parallel download of dc01 (16.87 GB)
│   ├── gen_paths.sbatch         # H200 array job — sharded path generation
│   ├── merge_shards.sbatch      # CPU node — concat + per-shard JSON merge
│   └── run_notebooks.sbatch     # H200 — execute all paper notebooks
├── code/                        # gen_dataset.py + utils.py with the RIS/dtype/skip patches
├── compat/
│   ├── _drjit1x_compat.py       # runtime shim
│   └── apply_sionna_patches.py  # in-place source patches
├── notebooks/                   # Synthetic_Data, ITU/Learned/Neural Materials, CDFs, Heat_Maps, CIRs
├── data/                        # tfrecords land here; coordinates.csv + spec.json shipped
├── scenes/inue_simple/          # Mitsuba scene
└── checkpoints/, results/       # populated by training notebooks
```

## NYU torch resources used

| Resource | Path / value |
|---|---|
| Account | `torch_pr_100_tandon_advanced` |
| Container image | `/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif` |
| Overlay template | `/scratch/work/public/overlay-fs-ext3/overlay-15GB-500K.ext3.gz` |
| Project dir | `/scratch/$USER/ho_rt` |
| Overlay path | `/scratch/$USER/ho_rt/horp-overlay.ext3` |
| GPU (where applicable) | `--gres=gpu:h200:1` |

## Why the patches

The repo originally targets **Sionna 0.18 + DrJit 0.4 + Mitsuba 3.5**, which
links against the OptiX 7.6 ABI. NVIDIA driver 570+ ships OptiX 8 — the older
DrJit can no longer create a pipeline (`OPTIX_ERROR_PIPELINE_LINK_ERROR 7251`).

Solution: **upgrade DrJit/Mitsuba to 1.x/3.8** (which ship the OptiX 8 ABI) and
patch Sionna 0.18 in five small places to bridge the API rename:

1. `dr.reinterpret_array_v` ↔ `dr.reinterpret_array` (rename only).
2. `mi.{Point2f,Point3f,Vector*f}` constructors now require `(C, N)` layout —
   wrap the 12 call sites with `_ls()` (auto-transpose) in `sionna/rt/utils.py`.
3. `mi_to_tf_tensor` transposes `.tf()` output back to legacy `(N, C)` layout.
4. `mi.Color0f(0.)` → `mi.Color0f()` (1.x rejects size-1 input on 0-channel type).
5. `prims_i = dr.select(active, offsets + si.prim_index, -1)` → cast both branches
   to `mi.Int32` so the `-1` sentinel doesn't fail UInt promotion.
6. (in `code/utils.py`) `serialize_traced_paths` strips the two RIS entries
   that Sionna 0.18 added to `trace_paths()` (8-tuple instead of 6-tuple) and
   casts every field to its expected dtype before `tf.io.serialize_tensor` so
   the deserializer's `parse_tensor(..., out_type=...)` matches the proto.

`compat/apply_sionna_patches.py` applies all these in-place; `setup_overlay.sbatch` runs it.

## End-to-end on torch

Run these from a torch login node. Replace `<JOBID>` with the numeric ID
printed by the previous `sbatch`. All chains use `--dependency=afterok:<id>`
so a failure in one stage cleanly cancels the rest.

```bash
# 0. Project dir
mkdir -p /scratch/$USER && cd /scratch/$USER
git clone https://github.com/tp1000d/ho_rt.git
cd ho_rt
mkdir -p logs

# 1. Build the overlay [~30-45 min on a CPU node]
sbatch slurm/setup_overlay.sbatch
# -> Submitted batch job <SETUP_JOBID>

# 2. Download dichasus-dc01 [~10 min on a CPU node, can run in parallel with step 1]
sbatch slurm/download_dichasus.sbatch
# -> Submitted batch job <DL_JOBID>

# 3. Path generation, 4-way sharded across H200s [~1-2 h each, run in parallel]
sbatch --dependency=afterok:<SETUP_JOBID>:<DL_JOBID> slurm/gen_paths.sbatch
# -> Submitted batch job <GEN_JOBID> (array; expands to GEN_JOBID_0..3)

# 4. Merge shards
sbatch --dependency=afterok:<GEN_JOBID> slurm/merge_shards.sbatch
# -> Submitted batch job <MERGE_JOBID>

# 5. Run all notebooks on a single H200 [~4-6 h]
sbatch --dependency=afterok:<MERGE_JOBID> slurm/run_notebooks.sbatch

squeue -u "$USER"
```

After the chain finishes, the executed notebooks (with all output cells in
place) are in `notebooks/`, training weights in `checkpoints/`, and figures
plus any new `*.pdf` files in `results/`.

## Resource sizing

| Stage | Node | Wall time | Why |
|-------|------|-----------|-----|
| Overlay build | CPU | ~30-45 min | apt + pip + miniconda + caches |
| Download dc01 | CPU | ~10 min | 16-stream parallel byte range |
| Path gen (4 shards × 2500) | 4 × H200 | ~1-2 h each | Sionna RT shoot-and-bounce |
| Notebook training | 1 × H200 | ~4-6 h | Adam over 10k traced paths |

H200 has 141 GB HBM3e; Sionna RT comfortably fits a 4M-sample shoot in <30 GB.

## References

- Paper: <https://arxiv.org/abs/2311.18558>
- Original code: <https://github.com/NVlabs/diff-rt-calibration>
- DICHASUS: <https://dichasus.inue.uni-stuttgart.de/datasets/data/dichasus-dcxx/>
- DICHASUS DOI: <https://doi.org/10.18419/darus-3831>
- NYU HPC project portal: <https://projects.hpc.nyu.edu>

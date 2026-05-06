# ho_rt — Hoydis-style Differentiable Ray Tracing reproduction on NYU torch HPC

Reproduce *Learning Radio Environments by Differentiable Ray Tracing*
(Hoydis et al., NVIDIA, 2023) on NYU torch HPC H200 GPUs.

This bundle is a self-contained packaging of [NVlabs/diff-rt-calibration]
plus the patches needed to run on NVIDIA driver 570 (which ships OptiX 8 ABI),
plus SLURM submission scripts for torch.

```
ho_rt/
├── Singularity.def         # CUDA 12.4 + TF 2.15 + Sionna 0.18 + drjit 1.x + patches
├── slurm/
│   ├── build_singularity.sbatch  # CPU node, --fakeroot, builds horp.sif
│   ├── download_dichasus.sbatch  # CPU node, 16-stream parallel download of dc01
│   ├── gen_paths.sbatch          # GPU array job, sharded path generation
│   ├── merge_shards.sbatch       # CPU node, concatenate per-shard tfrecords
│   └── run_notebooks.sbatch      # GPU node, executes all paper notebooks
├── code/                   # gen_dataset.py + utils.py with the RIS/dtype/skip patches
├── compat/
│   ├── _drjit1x_compat.py       # runtime shim
│   └── apply_sionna_patches.py  # in-place source patches (run during image build)
├── notebooks/              # Synthetic_Data, ITU/Learned/Neural Materials, CDFs, etc.
├── data/                   # tfrecords land here; coordinates.csv + spec.json shipped
├── scenes/                 # inue_simple Mitsuba scene
└── checkpoints/, results/  # populated by training notebooks
```

## Why the patches

The repo originally targets **Sionna 0.18 + DrJit 0.4 + Mitsuba 3.5**, which
links against the OptiX 7.6 ABI. NVIDIA driver 570+ ships OptiX 8 — the older
DrJit can no longer create a pipeline (`OPTIX_ERROR_PIPELINE_LINK_ERROR 7251`).

Solution used here: **upgrade DrJit/Mitsuba to 1.x/3.8** (which ship the OptiX 8
ABI) and patch Sionna 0.18 in five small places to bridge the API rename:

1. `dr.reinterpret_array_v` ↔ `dr.reinterpret_array` (rename only).
2. `mi.{Point2f,Point3f,Vector*f}` constructors now require `(C, N)` layout —
   wrap the 12 call sites with `_ls()` (auto-transpose) in `sionna/rt/utils.py`.
3. `mi_to_tf_tensor` transposes `.tf()` output back to legacy `(N, C)` layout.
4. `mi.Color0f(0.)` → `mi.Color0f()` (1.x rejects size-1 input on a 0-channel type).
5. `prims_i = dr.select(active, offsets + si.prim_index, -1)` → cast both branches
   to `mi.Int32` so the `-1` sentinel doesn't fail UInt promotion.
6. (in `code/utils.py`) `serialize_traced_paths` now strips the two RIS entries
   that Sionna 0.18 added to `trace_paths()` (8-tuple instead of 6-tuple), and
   casts every field to its expected dtype before `tf.io.serialize_tensor` so
   the deserializer's `parse_tensor(..., out_type=...)` matches the proto.

All patches are applied automatically by `compat/apply_sionna_patches.py`,
which the Singularity build invokes once.

## End-to-end on torch

```bash
ssh torch  # once, to populate the ControlMaster socket (8h persist)
cd $SCRATCH
git clone https://github.com/tp1000d/ho_rt.git
cd ho_rt
mkdir -p logs

# (1) Build the container [~30 min on a CPU node]
sbatch slurm/build_singularity.sbatch

# (2) Download dichasus-dc01 [~10 min on a CPU node]
sbatch slurm/download_dichasus.sbatch

# Wait for both to finish
squeue -u $USER

# (3) Path generation, 4-way sharded across H200 GPUs [~1-2 h each]
sbatch slurm/gen_paths.sbatch       # array 0..3, %4 concurrency

# (4) Once all shards finish, merge them
sbatch --dependency=afterok:<gen_paths_jobid> slurm/merge_shards.sbatch

# (5) Run the four training + three figure notebooks [~4-6 h on H200]
sbatch --dependency=afterok:<merge_jobid> slurm/run_notebooks.sbatch
```

## Resource sizing

| Stage | Node | Wall time | Why |
|-------|------|-----------|-----|
| Build SIF | CPU | ~30 min | apt + pip + cache |
| Download dc01 | CPU | ~10 min | 16-stream parallel byte range |
| Path gen (4 shards × 2500) | 4 × H200 | ~1-2 h each | Sionna RT shoot-and-bounce |
| Notebook training | 1 × H200 | ~4-6 h | Adam over 5k traced paths × 6k steps |

H200 has 141 GB HBM3e; Sionna RT comfortably fits a 4M-sample shoot in <30 GB.
The path-gen job batches one receiver position at a time so memory isn't the
bottleneck — speed is dominated by OptiX raycasting throughput.

## References

- Paper: <https://arxiv.org/abs/2311.18558>
- Original code: <https://github.com/NVlabs/diff-rt-calibration>
- DICHASUS: <https://dichasus.inue.uni-stuttgart.de/datasets/data/dichasus-dcxx/>
- DICHASUS DOI: <https://doi.org/10.18419/darus-3831>

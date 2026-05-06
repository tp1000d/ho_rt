"""
Re-pad ONE shard of traced-paths tfrecords to a specified global max_num_paths_*.

Why: gen_dataset.py post-processing pads each shard's records to that shard's
*own* per-shard max. When you concatenate multiple shards, the resulting
tfrecords file mixes records of different shapes, and tf.data.batch() dies:

    Cannot batch tensors with different shapes in component 3.

Fix: re-pad every shard up to a single agreed-on global max, then concatenate.

Usage:
    python repad_traced_paths.py \
        --in dichasus-dc01-0.tfrecords \
        --in_json dichasus-dc01-0.json \
        --out dichasus-dc01-0.repad.tfrecords \
        --max_spec 198 --max_diff 6 --max_scat 448 \
        --gpu 0

Run two of these in parallel (one per GPU) for the two-shard layout this
project produces.
"""
import argparse
import json
import os
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--in", dest="inp", required=True,
                    help="Input shard tfrecords (relative to ../data/traced_paths/)")
parser.add_argument("--in_json", required=True,
                    help="Input shard JSON (relative to ../data/traced_paths/)")
parser.add_argument("--out", required=True,
                    help="Output tfrecords (relative to ../data/traced_paths/)")
parser.add_argument("--max_spec", type=int, required=True)
parser.add_argument("--max_diff", type=int, required=True)
parser.add_argument("--max_scat", type=int, required=True)
parser.add_argument("--gpu", type=int, default=0)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
for g in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except RuntimeError as e:
        print(e)
tf.get_logger().setLevel("ERROR")

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CODE_DIR)
DATA_DIR = os.path.normpath(os.path.join(CODE_DIR, "..", "data", "traced_paths"))

# init_scene loads via "../scenes/...", so we must run from code/ for the scene
# import. After the scene is loaded we cd to data/ for tfrecords I/O.
os.chdir(CODE_DIR)
from utils import (init_scene, place_transmitter_arrays, instantiate_receivers,
                   deserialize_paths_as_tensor_dicts, tensor_dicts_to_traced_paths,
                   pad_traced_paths, serialize_traced_paths)

scene = init_scene("inue_simple", use_tx_array=True)
place_transmitter_arrays(scene, [1, 2])
instantiate_receivers(scene, 1)

os.chdir(DATA_DIR)

with open(args.in_json) as fh:
    in_meta = json.load(fh)
shard_size = in_meta.get("traced_paths_dataset_size", 0)
print(f"[GPU {args.gpu}] Re-padding {args.inp} ({shard_size} records) "
      f"to spec={args.max_spec} diff={args.max_diff} scat={args.max_scat}")

out_tmp = args.out + ".inprogress"
ds = tf.data.TFRecordDataset([args.inp]).map(deserialize_paths_as_tensor_dicts)
writer = tf.io.TFRecordWriter(out_tmp)

t0 = time.time()
n = 0
for record in ds:
    rx_pos, h_meas, traced = record[0], record[1], record[2:]
    traced = tensor_dicts_to_traced_paths(scene, traced)
    traced = pad_traced_paths(traced, args.max_spec, args.max_diff, args.max_scat)
    bytes_ = serialize_traced_paths(rx_pos, h_meas, traced, False)
    writer.write(bytes_)
    n += 1
    if n % 250 == 0:
        elapsed = time.time() - t0
        rate = n / max(elapsed, 1e-3)
        eta = (max(shard_size, n + 1) - n) / max(rate, 1e-3)
        print(f"[GPU {args.gpu}] {n}/{shard_size} ({rate:.1f}/s, eta {eta/60:.1f} min)",
              flush=True)
writer.close()
os.replace(out_tmp, args.out)
print(f"[GPU {args.gpu}] Wrote {args.out} ({os.path.getsize(args.out)} bytes, {n} records)",
      flush=True)

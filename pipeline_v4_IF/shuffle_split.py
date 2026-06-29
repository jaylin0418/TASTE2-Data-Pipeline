"""
Shuffle all IF parquet samples and re-split into train/dev.
Uses large_binary for audio column to avoid 2GB int32 offset overflow.
"""
import argparse
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import glob, os, collections

parser = argparse.ArgumentParser(description="Shuffle IF parquet and re-split into train/dev")
parser.add_argument("--input-dir",  required=True, dest="input_dir",
                    help="Directory of if_part_*.parquet files (output of to_parquet_if.py)")
parser.add_argument("--output-dir", required=True, dest="output_dir",
                    help="Directory to write shuffled_train_part_*.parquet and shuffled_dev.parquet")
parser.add_argument("--rows-per-file", type=int, default=4000, dest="rows_per_file")
parser.add_argument("--dev-size",      type=int, default=4000, dest="dev_size")
parser.add_argument("--seed",          type=int, default=42)
args = parser.parse_args()

INPUT_DIR    = args.input_dir
OUTPUT_DIR   = args.output_dir
ROWS_PER_FILE = args.rows_per_file
DEV_SIZE     = args.dev_size
SEED         = args.seed

os.makedirs(OUTPUT_DIR, exist_ok=True)

def to_large_binary_schema(table):
    """Cast all binary columns to large_binary."""
    arrays, fields = [], []
    for i, field in enumerate(table.schema):
        col = table.column(i)
        if pa.types.is_binary(field.type):
            col = col.cast(pa.large_binary())
            field = pa.field(field.name, pa.large_binary(), nullable=field.nullable)
        arrays.append(col)
        fields.append(field)
    return pa.table(dict(zip([f.name for f in fields], arrays)))

files = sorted(glob.glob(f"{INPUT_DIR}/if_part_*.parquet"))
print(f"Found {len(files)} parquet files")

# Load and cast all tables
tables = []
row_counts = []
for f in files:
    t = pq.read_table(f)
    t = to_large_binary_schema(t)
    tables.append(t)
    row_counts.append(len(t))
    print(f"  {os.path.basename(f)}: {len(t)} rows")

total = sum(row_counts)
print(f"Total rows: {total}")

# Build global → (table_idx, local_row) map
global_to_local = []
for t_idx, rc in enumerate(row_counts):
    for r_idx in range(rc):
        global_to_local.append((t_idx, r_idx))

# Shuffle
rng = np.random.default_rng(SEED)
perm = rng.permutation(total)
dev_global = perm[:DEV_SIZE].tolist()
train_global = perm[DEV_SIZE:].tolist()
print(f"Train: {len(train_global)}, Dev: {len(dev_global)}")

def write_chunk(global_indices, out_path):
    from collections import defaultdict
    # Group by source table
    table_row_lists = defaultdict(list)
    order = []
    for g in global_indices:
        t_idx, r_idx = global_to_local[g]
        pos = len(table_row_lists[t_idx])
        table_row_lists[t_idx].append(r_idx)
        order.append((t_idx, pos))

    taken = {t_idx: tables[t_idx].take(rows) for t_idx, rows in table_row_lists.items()}

    rows = [taken[t_idx].slice(pos, 1) for t_idx, pos in order]
    result = pa.concat_tables(rows)  # all large_binary, safe to concat
    pq.write_table(result, out_path)
    return result

# Write dev
print("\nWriting dev...")
dev_table = write_chunk(dev_global, f"{OUTPUT_DIR}/shuffled_dev.parquet")
metas = dev_table['meta'].to_pylist()
dims = collections.Counter(m['para_dimension'] for m in metas)
print(f"Dev para_dimension: {dict(dims)}")
print(f"Written: shuffled_dev.parquet ({len(dev_table)} rows)")

# Write train
n_train = len(train_global)
n_parts = (n_train + ROWS_PER_FILE - 1) // ROWS_PER_FILE
print(f"\nWriting {n_parts} train files...")
for i in range(n_parts):
    start = i * ROWS_PER_FILE
    end = min(start + ROWS_PER_FILE, n_train)
    chunk = write_chunk(train_global[start:end], f"{OUTPUT_DIR}/shuffled_train_part_{i:04d}.parquet")
    print(f"  Part {i:04d}: {len(chunk)} rows")

print("\nDone.")

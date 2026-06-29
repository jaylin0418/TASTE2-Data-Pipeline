"""
Build a second, TRAINING-READY version of the TASTE-IF-SFT-48K dataset:
keeps the original `message[].audio` (binary) field expected directly by
the TASTE SFT data loader (sft_processor.py), while applying the same
cleanups as the HF-viewer version (build_hf_dataset_with_audio.py):
  - add a top-level `system_prompt` column
  - rename meta.para_dimension -> meta.instruction_type, drop empty
    topic/scenario_id/scenario_description fields
  - fix mislabeled user-turn para_tags (always null, since instruction
    audio is always synthesized neutrally)

No top-level instruction_audio/response_audio columns (those are only
needed for the HF Dataset Viewer audio player), so this stays in the
exact shape sft_processor.py expects.
"""
import os
import glob
import pyarrow as pa
import pyarrow.parquet as pq

SRC_DIR = "/work/jaylin0418/IF_data_generation/shuffled_output"
DST_DIR = "/work/jaylin0418/IF_data_generation/shuffled_output_train"
ROW_GROUP_SIZE = 300

SYSTEM_PROMPT = (
    "You are a helpful spoken assistant. Listen to the user's spoken instruction "
    "and respond with only the requested spoken content, following the "
    "instruction exactly."
)

MESSAGE_STRUCT = pa.list_(pa.struct({
    "role": pa.string(),
    "text": pa.string(),
    "audio": pa.binary(),
    "timestamp_range": pa.list_(pa.int64()),
    "para_tags": pa.string(),
}))

SCHEMA = pa.schema([
    ("idx", pa.string()),
    ("system_prompt", pa.string()),
    ("meta", pa.struct({
        "instruction_type": pa.string(),
    })),
    ("message", MESSAGE_STRUCT),
])


def process_file(src_path, dst_path):
    pf = pq.ParquetFile(src_path)
    writer = pq.ParquetWriter(dst_path, SCHEMA)
    for batch in pf.iter_batches(batch_size=ROW_GROUP_SIZE):
        df = batch.to_pandas()
        rows = []
        for i in range(len(df)):
            row = df.loc[i]
            messages = row["message"]
            if hasattr(messages, "tolist"):
                messages = messages.tolist()
            messages = [dict(m) for m in messages]
            for m in messages:
                tr = m.get("timestamp_range")
                if hasattr(tr, "tolist"):
                    m["timestamp_range"] = tr.tolist()
            if messages and messages[0].get("role") == "user":
                messages[0]["para_tags"] = None
            rows.append({
                "idx": row["idx"],
                "system_prompt": SYSTEM_PROMPT,
                "meta": {"instruction_type": row["meta"]["para_dimension"]},
                "message": messages,
            })
        table = pa.Table.from_pylist(rows, schema=SCHEMA)
        writer.write_table(table, row_group_size=ROW_GROUP_SIZE)
    writer.close()


if __name__ == "__main__":
    os.makedirs(DST_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(SRC_DIR, "*.parquet")))
    for src in files:
        dst = os.path.join(DST_DIR, os.path.basename(src))
        print("Processing", src, "->", dst)
        process_file(src, dst)
        f = pq.ParquetFile(dst)
        sizes = [f.metadata.row_group(i).total_byte_size for i in range(f.metadata.num_row_groups)]
        print("  row_groups:", f.metadata.num_row_groups, "max_rg_bytes:", max(sizes))

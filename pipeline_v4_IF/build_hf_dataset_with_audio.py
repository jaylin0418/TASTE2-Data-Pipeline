"""
Rebuild the TASTE-IF-SFT-48K parquet shards with:
  - a top-level `system_prompt` column (same string for every row)
  - top-level `instruction_audio` / `response_audio` columns stored as
    struct{bytes, path} (the standard HF `Audio()` storage layout) so the
    Dataset Viewer can play them, declared via README dataset_info YAML
while keeping the original `idx`, `meta`, `message` structure unchanged
(so it stays compatible with the TASTE SFT data loader).

We avoid `datasets.Dataset.from_generator` + `Audio()` encoding because it
requires `torchcodec`, which is incompatible with the installed torch 2.3.1.
Instead we build the Arrow table directly with pyarrow.
"""
import os
import glob
import pyarrow as pa
import pyarrow.parquet as pq

SRC_DIR = "/work/jaylin0418/IF_data_generation/shuffled_output_v3"
DST_DIR = "/work/jaylin0418/IF_data_generation/shuffled_output_v4"
ROW_GROUP_SIZE = 300

SYSTEM_PROMPT = (
    "You are a helpful spoken assistant. Listen to the user's spoken instruction "
    "and respond with only the requested spoken content, following the "
    "instruction exactly."
)

AUDIO_STRUCT = pa.struct({"bytes": pa.binary(), "path": pa.string()})

MESSAGE_STRUCT = pa.list_(pa.struct({
    "role": pa.string(),
    "text": pa.string(),
    "timestamp_range": pa.list_(pa.int64()),
    "para_tags": pa.string(),
}))

SCHEMA = pa.schema([
    ("idx", pa.string()),
    ("system_prompt", pa.string()),
    ("meta", pa.struct({
        "instruction_type": pa.string(),
    })),
    ("instruction_audio", AUDIO_STRUCT),
    ("response_audio", AUDIO_STRUCT),
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
            # User-turn audio is always synthesized with a neutral/calm voice
            # (see tts_if_data.py), so any para_tags on the user turn are
            # mislabeled metadata leftover from the shared style dict.
            if messages and messages[0].get("role") == "user":
                messages[0]["para_tags"] = None
            rows.append({
                "idx": row["idx"],
                "system_prompt": row["system_prompt"],
                "meta": {"instruction_type": row["meta"]["instruction_type"]},
                "instruction_audio": dict(row["instruction_audio"]),
                "response_audio": dict(row["response_audio"]),
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

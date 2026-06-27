#!/usr/bin/env python3
"""
hf_to_parquet.py – Convert HuggingFace arrow dataset to to_parquet.py format.

Input:  /work/.../dataset_patched/  (arrow files, one row per turn)
Output: /work/.../parquet_sft/      (parquet files, one row per conversation)

Each output row matches to_parquet.py's SCHEMA:
  idx     : conversation_id
  meta    : {para_dimension, topic, scenario_id, scenario_description}
  message : [{role, text, audio (bytes), timestamp_range, para_tags}]
"""

from __future__ import annotations

import io
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

CHUNK_BYTES = 1 * 1024 ** 3  # ~1 GB per parquet file

# ── Schema (same as to_parquet.py) ───────────────────────────────────────────
_MSG_TYPE = pa.struct([
    pa.field("role",            pa.string()),
    pa.field("text",            pa.string()),
    pa.field("audio",           pa.binary(),          nullable=True),
    pa.field("timestamp_range", pa.list_(pa.int64()), nullable=True),
    pa.field("para_tags",       pa.string(),          nullable=True),
])

_META_TYPE = pa.struct([
    pa.field("para_dimension",       pa.string()),
    pa.field("topic",                pa.string()),
    pa.field("scenario_id",          pa.string()),
    pa.field("scenario_description", pa.string()),
])

SCHEMA = pa.schema([
    pa.field("idx",     pa.string()),
    pa.field("meta",    _META_TYPE),
    pa.field("message", pa.list_(_MSG_TYPE)),
])


# ── Helpers ───────────────────────────────────────────────────────────────────
def build_para_tags(paralinguistic_info, emotion=None) -> Optional[str]:
    tags = {}
    if isinstance(paralinguistic_info, dict):
        tags.update({k: v for k, v in paralinguistic_info.items() if v is not None})
    # emotion column takes priority if paralinguistic_info.emotion is missing
    if emotion is not None and "emotion" not in tags:
        tags["emotion"] = emotion
    return json.dumps(tags, ensure_ascii=False) if tags else None


def build_row(conv_id: str, turns: List[dict]) -> tuple[dict, int]:
    first = turns[0]
    meta = {
        "para_dimension":       first.get("mode") or "normal",
        "topic":                first.get("topic") or "",
        "scenario_id":          f"scenario_{conv_id}",
        "scenario_description": first.get("scenario") or "",
    }

    messages = []
    approx_bytes = 0

    for turn in turns:
        speaker = turn.get("speaker") or "user"
        role = "user" if speaker.lower() == "user" else "assistant"

        audio_bytes: Optional[bytes] = None
        audio_struct = turn.get("audio")
        if isinstance(audio_struct, dict):
            audio_bytes = audio_struct.get("bytes")
        elif isinstance(audio_struct, bytes):
            audio_bytes = audio_struct

        start_s = turn.get("audio_start") or 0.0
        end_s   = turn.get("audio_end")   or 0.0
        ts = [int(start_s * 1000), int(end_s * 1000)] if audio_bytes else None

        messages.append({
            "role":            role,
            "text":            turn.get("text") or "",
            "audio":           audio_bytes,
            "timestamp_range": ts,
            "para_tags":       build_para_tags(turn.get("paralinguistic_info"), turn.get("emotion")),
        })
        approx_bytes += len(audio_bytes) if audio_bytes else 0

    approx_bytes += 512
    return {"idx": conv_id, "meta": meta, "message": messages}, approx_bytes


# ── Writer ────────────────────────────────────────────────────────────────────
class ChunkedWriter:
    def __init__(self, out_dir: Path, prefix: str, chunk_bytes: int = CHUNK_BYTES):
        self.out_dir     = out_dir
        self.prefix      = prefix
        self.chunk_bytes = chunk_bytes
        out_dir.mkdir(parents=True, exist_ok=True)
        self._part  = 0
        self._rows: List[dict] = []
        self._bytes = 0

    def add(self, row: dict, row_bytes: int):
        self._rows.append(row)
        self._bytes += row_bytes
        if self._bytes >= self.chunk_bytes:
            self._flush()

    def _flush(self):
        if not self._rows:
            return
        path     = self.out_dir / f"{self.prefix}_part_{self._part:04d}.parquet"
        tmp_path = self.out_dir / f"{self.prefix}_part_{self._part:04d}_{os.getpid()}.parquet.tmp"
        table    = pa.Table.from_pylist(self._rows, schema=SCHEMA)
        pq.write_table(table, tmp_path, compression="snappy")
        tmp_path.rename(path)
        log.info("Wrote %s  (%d conversations, %.2f MB)", path, len(self._rows), self._bytes / 1e6)
        self._rows  = []
        self._bytes = 0
        self._part += 1

    def close(self):
        self._flush()


# ── Main ──────────────────────────────────────────────────────────────────────
def convert(data_dir: str, out_dir: str, prefix: str = "sft"):
    data_path = Path(data_dir)
    arrow_files = sorted(data_path.glob("*.arrow"))
    if not arrow_files:
        log.error("No .arrow files found in %s", data_dir)
        return

    log.info("Found %d arrow files", len(arrow_files))

    writer = ChunkedWriter(Path(out_dir), prefix)

    # Buffer conversations across arrow files
    # (turns of the same conversation_id may span multiple arrow files)
    pending: Dict[str, List[dict]] = defaultdict(list)
    current_conv_order: List[str] = []  # track insertion order

    for arrow_file in arrow_files:
        log.info("Reading %s", arrow_file.name)
        with ipc.open_stream(str(arrow_file)) as reader:
            table = reader.read_all()

        rows = table.to_pylist()

        for row in rows:
            conv_id = row["conversation_id"]
            if conv_id not in pending:
                current_conv_order.append(conv_id)
            pending[conv_id].append(row)

        # Flush complete conversations (those whose conv_id won't appear again in future files)
        # Strategy: flush all except the last conv_id seen (it may continue in next file)
        if current_conv_order:
            last_conv = current_conv_order[-1]
            for conv_id in list(current_conv_order[:-1]):
                turns = sorted(pending.pop(conv_id), key=lambda x: x.get("turn_index", 0))
                row, rb = build_row(conv_id, turns)
                writer.add(row, rb)
            current_conv_order = [last_conv]

    # Flush remaining
    for conv_id in current_conv_order:
        if conv_id in pending:
            turns = sorted(pending.pop(conv_id), key=lambda x: x.get("turn_index", 0))
            row, rb = build_row(conv_id, turns)
            writer.add(row, rb)

    writer.close()
    log.info("Done. Parquet written to: %s", out_dir)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, help="Path to dataset_patched folder")
    p.add_argument("--out-dir",  required=True, help="Output parquet folder")
    p.add_argument("--prefix",   default="sft", help="Parquet filename prefix")
    args = p.parse_args()
    convert(args.data_dir, args.out_dir, args.prefix)

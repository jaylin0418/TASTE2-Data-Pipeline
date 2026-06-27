 #!/usr/bin/env python3
"""
to_parquet_variant.py – Convert syn_variant outputs to chunked Parquet files.

Input layout
------------
<variant_root>/variant/<topic>/wav/<scenario>/individual/turn_metadata.json

Output layout
-------------
<out_dir>/variant_part_0000.parquet, ...

Each row matches to_parquet.py SCHEMA:
  idx     : "<topic>/<scenario>"
  meta    : {para_dimension="variant", topic, scenario_id, scenario_description=""}
  message : [{role, text, audio (bytes), timestamp_range, para_tags}]
"""
from __future__ import annotations

import io
import json
import logging
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
import torchaudio

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

CHUNK_BYTES = 1 * 1024 ** 3

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


def load_wav_bytes(wav_path: Path) -> Optional[bytes]:
    if not wav_path.exists():
        log.warning("WAV not found: %s", wav_path)
        return None
    try:
        tensor, sr = torchaudio.load(str(wav_path))
        buf = io.BytesIO()
        torchaudio.save(buf, tensor, sr, format="wav")
        return buf.getvalue()
    except Exception as exc:
        log.warning("Cannot load %s: %s", wav_path, exc)
        return None


def build_row(
    topic: str,
    scenario_id: str,
    turns: List[dict],
    system_prompt: Optional[str] = None,
) -> Tuple[dict, int]:
    meta = {
        "para_dimension":       "variant",
        "topic":                topic,
        "scenario_id":          scenario_id,
        "scenario_description": "",
    }

    messages: List[dict] = []
    approx_bytes = 0

    if system_prompt:
        messages.append({
            "role":            "system",
            "text":            system_prompt,
            "audio":           None,
            "timestamp_range": None,
            "para_tags":       None,
        })

    cumulative_ms = 0
    for turn in turns:
        role = "user" if turn["role"] == "User" else "assistant"
        text = turn.get("text", "")

        audio_path = Path(turn["audio_path"])
        audio = load_wav_bytes(audio_path)

        # para_tags from variant turn_metadata
        para_tags = turn.get("para_tags") or {}
        variant_type = turn.get("variant_type")
        if variant_type and "emotion" not in para_tags:
            # e.g. "emotion_surprised" -> extract emotion value
            if variant_type.startswith("emotion_"):
                para_tags["emotion"] = variant_type[len("emotion_"):]
            else:
                para_tags["variant_type"] = variant_type

        para_tags_str: Optional[str] = None
        non_null = {k: v for k, v in para_tags.items() if v is not None}
        if non_null:
            para_tags_str = json.dumps(non_null)

        duration_ms = int(turn.get("audio_duration", 0.0) * 1000)
        start_ms = cumulative_ms
        end_ms = cumulative_ms + duration_ms
        cumulative_ms = end_ms

        messages.append({
            "role":            role,
            "text":            text,
            "audio":           audio,
            "timestamp_range": [start_ms, end_ms] if audio else None,
            "para_tags":       para_tags_str,
        })
        approx_bytes += len(audio) if audio else 0

    approx_bytes += 512
    idx = f"{topic}/{scenario_id}"
    return {"idx": idx, "meta": meta, "message": messages}, approx_bytes


class ChunkedParquetWriter:
    def __init__(self, out_dir: Path, prefix: str, chunk_bytes: int = CHUNK_BYTES):
        self.out_dir = out_dir
        self.prefix = prefix
        self.chunk_bytes = chunk_bytes
        out_dir.mkdir(parents=True, exist_ok=True)
        self._part = 0
        self._rows: List[dict] = []
        self._bytes = 0

    def add(self, row: dict, row_bytes: int) -> None:
        self._rows.append(row)
        self._bytes += row_bytes
        if self._bytes >= self.chunk_bytes:
            self._flush()

    def _flush(self) -> None:
        if not self._rows:
            return
        path = self.out_dir / f"{self.prefix}_part_{self._part:04d}.parquet"
        tmp = self.out_dir / f"{self.prefix}_part_{self._part:04d}_{os.getpid()}.parquet.tmp"
        table = pa.Table.from_pylist(self._rows, schema=SCHEMA)
        pq.write_table(table, tmp, compression="snappy")
        tmp.rename(path)
        log.info("Wrote %s  (%d rows, %.2f MB)", path, len(self._rows), self._bytes / 1e6)
        self._rows = []
        self._bytes = 0
        self._part += 1

    def close(self) -> None:
        self._flush()


def convert(variant_root: str, out_dir: str, prefix: str = "variant") -> None:
    root = Path(variant_root) / "variant"
    out_path = Path(out_dir)
    writer = ChunkedParquetWriter(out_path, prefix)

    total = 0
    for topic_dir in sorted(root.iterdir()):
        if not topic_dir.is_dir():
            continue
        topic = topic_dir.name
        wav_dir = topic_dir / "wav"
        if not wav_dir.exists():
            continue

        for scenario_dir in sorted(wav_dir.iterdir()):
            if not scenario_dir.is_dir():
                continue
            metadata_file = scenario_dir / "individual" / "turn_metadata.json"
            if not metadata_file.exists():
                log.warning("No turn_metadata.json: %s", scenario_dir)
                continue

            turns = json.loads(metadata_file.read_text("utf-8"))
            if not turns:
                continue

            scenario_id = scenario_dir.name
            # Load system prompt: topic_dir/txt/system_prompt_txt/Topic_scenarioN_system_prompt.txt
            system_prompt: Optional[str] = None
            n_match = re.search(r"scenario(\d+)", scenario_id)
            if n_match:
                sp_file = topic_dir / "txt" / "system_prompt_txt" / \
                    f"{topic}_scenario{n_match.group(1)}_system_prompt.txt"
                if sp_file.exists():
                    try:
                        system_prompt = sp_file.read_text("utf-8").strip() or None
                    except Exception:
                        pass

            row, row_bytes = build_row(topic, scenario_id, turns, system_prompt=system_prompt)
            writer.add(row, row_bytes)
            total += 1
            if total % 500 == 0:
                log.info("Processed %d dialogues...", total)

    writer.close()
    log.info("Done. Total dialogues: %d  Output: %s", total, out_dir)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--variant-root", required=True, help="syn_variant root (contains variant/ dir)")
    p.add_argument("--out-dir", required=True, help="Output parquet directory")
    p.add_argument("--prefix", default="variant", help="Parquet file prefix (default: variant)")
    args = p.parse_args()
    convert(args.variant_root, args.out_dir, args.prefix)

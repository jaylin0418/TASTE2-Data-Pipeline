#!/usr/bin/env python3
"""
to_parquet_if.py — Convert IF TTS outputs to chunked Parquet files for TASTE training.

Input layout
------------
<data_dir>/
└── <IF_type>/
    └── wav/
        └── <sample_id>/
            └── individual/
                ├── turn00.wav
                ├── turn01.wav
                └── turn_metadata.json

Output layout
-------------
<out_dir>/
├── if_part_0000.parquet
├── if_part_0001.parquet
└── ...

Schema matches Nano4/to_parquet.py so sft_processor can read it directly.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

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
        return wav_path.read_bytes()
    except Exception as exc:
        log.warning("Cannot read %s: %s", wav_path, exc)
        return None


def build_row(sample_id: str, if_type: str, turns: List[dict], wav_base: Path) -> Tuple[dict, int]:
    meta = {
        "para_dimension":       if_type,
        "topic":                "",
        "scenario_id":          "",
        "scenario_description": "",
    }
    messages: List[dict] = []
    approx_bytes = 0

    for turn in turns:
        role = "user" if turn["role"] == "User" else "assistant"
        text = turn.get("text", "")
        para_tags = turn.get("para_tags")
        start_ms = int(turn.get("audio_start", 0.0) * 1000)
        end_ms   = int(turn.get("audio_end",   0.0) * 1000)

        turn_wav = wav_base / f"turn{turn['turn_idx']:02d}.wav"
        audio = load_wav_bytes(turn_wav)

        para_tags_str: Optional[str] = None
        if para_tags:
            non_null = {k: v for k, v in para_tags.items() if v is not None}
            if non_null:
                para_tags_str = json.dumps(non_null)

        messages.append({
            "role":            role,
            "text":            text,
            "audio":           audio,
            "timestamp_range": [start_ms, end_ms] if audio else None,
            "para_tags":       para_tags_str,
        })
        approx_bytes += len(audio) if audio else 0

    approx_bytes += 512
    return {"idx": sample_id, "meta": meta, "message": messages}, approx_bytes


class ChunkedParquetWriter:
    def __init__(self, out_dir: Path, prefix: str, chunk_bytes: int = CHUNK_BYTES):
        self.out_dir = out_dir
        self.prefix = prefix
        self.chunk_bytes = chunk_bytes
        out_dir.mkdir(parents=True, exist_ok=True)
        self._part = 0
        self._rows: List[dict] = []
        self._bytes = 0
        self._total = 0

    def add(self, row: dict, row_bytes: int) -> None:
        self._rows.append(row)
        self._bytes += row_bytes
        self._total += 1
        if self._bytes >= self.chunk_bytes:
            self._flush()

    def _flush(self) -> None:
        if not self._rows:
            return
        import os
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
        log.info("Total rows written: %d", self._total)


def convert(data_dir: str, out_dir: str, chunk_bytes: int = CHUNK_BYTES,
            if_types: Optional[List[str]] = None, prefix: str = "if") -> None:
    data_path = Path(data_dir)
    out_path = Path(out_dir)
    type_filter = set(if_types) if if_types else None

    writer = ChunkedParquetWriter(out_path, prefix, chunk_bytes)
    processed = 0
    skipped = 0

    for type_dir in sorted(data_path.iterdir()):
        if not type_dir.is_dir():
            continue
        if_type = type_dir.name
        if type_filter and if_type not in type_filter:
            log.info("Skipping IF_type '%s'", if_type)
            continue

        wav_dir = type_dir / "wav"
        if not wav_dir.exists():
            log.warning("No wav dir: %s", type_dir)
            continue

        log.info("Processing IF_type: %s", if_type)
        for sample_dir in sorted(wav_dir.iterdir()):
            if not sample_dir.is_dir():
                continue
            meta_file = sample_dir / "individual" / "turn_metadata.json"
            if not meta_file.exists():
                log.warning("Missing turn_metadata.json: %s", sample_dir)
                skipped += 1
                continue

            turns = json.loads(meta_file.read_text("utf-8"))
            if not turns:
                skipped += 1
                continue

            row, row_bytes = build_row(
                sample_id=sample_dir.name,
                if_type=if_type,
                turns=turns,
                wav_base=sample_dir / "individual",
            )
            writer.add(row, row_bytes)
            processed += 1
            if processed % 1000 == 0:
                log.info("Progress: %d processed, %d skipped", processed, skipped)

    writer.close()
    log.info("Done. %d rows written, %d skipped. Parquet: %s", processed, skipped, out_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Convert IF TTS outputs to Parquet for TASTE training")
    p.add_argument("--data-dir",  required=True,
                   help="Root TTS output directory")
    p.add_argument("--out-dir",   required=True,
                   help="Output directory for Parquet files")
    p.add_argument("--chunk-gb",  type=float, default=1.0,
                   help="Max GB per Parquet part (default: 1.0)")
    p.add_argument("--if-types",  nargs="+", default=None,
                   help="Only convert these IF_types (default: all)")
    p.add_argument("--prefix",    default="if",
                   help="Output filename prefix (default: if)")
    args = p.parse_args()

    convert(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        chunk_bytes=int(args.chunk_gb * 1024 ** 3),
        if_types=args.if_types,
        prefix=args.prefix,
    )

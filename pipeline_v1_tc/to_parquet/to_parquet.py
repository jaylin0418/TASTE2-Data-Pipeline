#!/usr/bin/env python3
"""
to_parquet.py  –  Convert TEST_syn_data outputs to chunked Parquet files.

Input layout
------------
<data_dir>/
├── scenarios/<topic>/scenarios.json
└── <para_dim>/<topic>/
    └── wav/<dialogue_id>/
        ├── full.wav
        └── individual/
            ├── turn00.wav, turn01.wav, ...
            └── turn_metadata.json          # primary data source

Output layout
-------------
<out_dir>/
├── part_0000.parquet
├── part_0001.parquet
└── ...

Each row
--------
{
  "idx":  "<dialogue_id>",
  "meta": {
    "para_dimension":       "age",
    "topic":                "Fitness",
    "scenario_id":          "scenario1",
    "scenario_description": "...",
  },
  "message": [
    {
      "role":            "user" | "assistant",
      "text":            "...",
      "audio":           <bytes> | None,
      "timestamp_range": [start_ms, end_ms] | None,
      "para_tags":       '{"age": "old", ...}' | None,   # JSON string
    },
    ...
  ]
}
"""

from __future__ import annotations

import io
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
import torchaudio

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ─── tunables ─────────────────────────────────────────────────────────────────
CHUNK_BYTES = 1 * 1024 ** 3   # flush a new parquet part every ~1 GB

# ─── PyArrow schema ────────────────────────────────────────────────────────────
_MSG_TYPE = pa.struct([
    pa.field("role",            pa.string()),
    pa.field("text",            pa.string()),
    pa.field("audio",           pa.binary(),          nullable=True),
    pa.field("timestamp_range", pa.list_(pa.int64()), nullable=True),
    pa.field("para_tags",       pa.string(),          nullable=True),  # JSON string
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


# ─── scenario loading ──────────────────────────────────────────────────────────

def load_scenarios(scen_file: Path) -> Dict[str, str]:
    """
    Load scenarios.json → dict mapping scenario_id → description.
    e.g. {"scenario1": "A new mother...", "scenario2": "..."}
    """
    if not scen_file.exists():
        log.warning("scenarios.json not found: %s", scen_file)
        return {}
    data = json.loads(scen_file.read_text("utf-8"))
    result: Dict[str, str] = {}
    for item in data.get("scenarios", []):
        for scen_id, scen_data in item.items():
            result[scen_id] = scen_data.get("description", "")
    return result


# ─── per-turn audio loading ────────────────────────────────────────────────────

def load_wav_bytes(wav_path: Path) -> Optional[bytes]:
    """Load a WAV file and return its raw bytes, or None on failure."""
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


# ─── row builder ──────────────────────────────────────────────────────────────

def build_row(
    dialogue_id:  str,
    para_dim:     str,
    topic:        str,
    scenario_id:  str,
    scenario_desc: str,
    turns:        List[dict],
    wav_base_dir: Path,
    system_prompt: Optional[str] = None,
) -> Tuple[dict, int]:
    """
    Build one Parquet row from turn_metadata.json entries.

    Each entry in `turns` is one element from turn_metadata.json:
      {turn_idx, role, text, para_tags, audio_path, audio_start, audio_end, ...}

    Returns (row_dict, approx_bytes).
    """
    meta = {
        "para_dimension":       para_dim,
        "topic":                topic,
        "scenario_id":          scenario_id,
        "scenario_description": scenario_desc,
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

    for turn in turns:
        role      = "user" if turn["role"] == "User" else "assistant"
        text      = turn.get("text", "")
        para_tags = turn.get("para_tags")          # dict or None
        start_s   = turn.get("audio_start", 0.0)
        end_s     = turn.get("audio_end",   0.0)

        # Individual turn WAV is in the same folder as turn_metadata.json
        turn_wav = wav_base_dir / f"turn{turn['turn_idx']:02d}.wav"
        audio = load_wav_bytes(turn_wav)

        start_ms = int(start_s * 1000)
        end_ms   = int(end_s   * 1000)

        # Encode para_tags as JSON string (None if all values are null/absent)
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

    approx_bytes += len(json.dumps(meta)) + len(dialogue_id) + 512

    return {"idx": dialogue_id, "meta": meta, "message": messages}, approx_bytes


# ─── chunked writer ───────────────────────────────────────────────────────────

class ChunkedParquetWriter:
    def __init__(self, out_dir: Path, prefix: str, chunk_bytes: int = CHUNK_BYTES):
        self.out_dir     = out_dir
        self.prefix      = prefix
        self.chunk_bytes = chunk_bytes
        out_dir.mkdir(parents=True, exist_ok=True)
        self._part  = 0
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
        import os
        path = self.out_dir / f"{self.prefix}_part_{self._part:04d}.parquet"
        tmp_path = self.out_dir / f"{self.prefix}_part_{self._part:04d}_{os.getpid()}.parquet.tmp"
        table = pa.Table.from_pylist(self._rows, schema=SCHEMA)
        pq.write_table(table, tmp_path, compression="snappy")
        tmp_path.rename(path)  # atomic on same filesystem
        log.info(
            "Wrote %s  (%d rows, %.2f MB)",
            path, len(self._rows), self._bytes / 1e6,
        )
        self._rows  = []
        self._bytes = 0
        self._part += 1

    def close(self) -> None:
        self._flush()


# ─── entry point ─────────────────────────────────────────────────────────────

def convert(
    data_dir: str,
    out_dir: str,
    chunk_bytes: int = CHUNK_BYTES,
    para_dims: Optional[List[str]] = None,
    prefix_override: Optional[str] = None,
) -> None:
    """
    Walk <data_dir>/<para_dim>/<topic>/wav/<dialogue_id>/individual/turn_metadata.json
    and write Parquet files into a single flat <out_dir>/.

    Files are named <para_dim>_part_0000.parquet — para_dims are never mixed,
    but topics within the same para_dim are combined freely.

    If `para_dims` is given, only those para_dim names are processed.
    """
    data_path = Path(data_dir)
    out_path  = Path(out_dir)
    para_dim_filter = set(para_dims) if para_dims else None
    scenario_cache: Dict[str, Dict[str, str]] = {}
    writers: Dict[str, ChunkedParquetWriter] = {}

    for para_dim_dir in sorted(data_path.iterdir()):
        if not para_dim_dir.is_dir() or para_dim_dir.name == "scenarios":
            continue
        para_dim = para_dim_dir.name
        if para_dim_filter and para_dim not in para_dim_filter:
            log.info("Skipping para_dim '%s' (not in --para-dims list)", para_dim)
            continue

        for topic_dir in sorted(para_dim_dir.iterdir()):
            if not topic_dir.is_dir():
                continue
            topic = topic_dir.name

            wav_dir = topic_dir / "wav"
            if not wav_dir.exists():
                log.warning("wav dir not found: %s – skipping", wav_dir)
                continue

            if topic not in scenario_cache:
                scen_file = data_path / "scenarios" / topic / "scenarios.json"
                scenario_cache[topic] = load_scenarios(scen_file)
            scenarios = scenario_cache[topic]

            # one writer per para_dim, shared across all topics
            writer_key = prefix_override or para_dim
            if writer_key not in writers:
                writers[writer_key] = ChunkedParquetWriter(out_path, writer_key, chunk_bytes)
            writer = writers[writer_key]

            for dialogue_dir in sorted(wav_dir.iterdir()):
                if not dialogue_dir.is_dir():
                    continue
                dialogue_id   = dialogue_dir.name
                metadata_file = dialogue_dir / "individual" / "turn_metadata.json"

                if not metadata_file.exists():
                    log.warning("No turn_metadata.json in %s – skipping", dialogue_dir)
                    continue

                turns = json.loads(metadata_file.read_text("utf-8"))
                if not turns:
                    log.warning("Empty turn_metadata.json: %s – skipping", metadata_file)
                    continue

                m = re.search(r"_(\d+)_\d+$", dialogue_id)
                scenario_id   = f"scenario{m.group(1)}" if m else "scenario1"
                scenario_desc = scenarios.get(scenario_id, "")

                # Load system prompt: TC style (dialogue_dir/system_prompt.txt)
                # or TC_emo style (topic_dir/txt/system_prompt_txt/Topic_scenarioN_system_prompt.txt)
                system_prompt: Optional[str] = None
                sp_file = dialogue_dir / "system_prompt.txt"
                if not sp_file.exists():
                    n_match = re.search(r"_(\d+)$", dialogue_id)
                    if n_match:
                        topic_clean = topic.replace("_emo", "")
                        sp_file = topic_dir / "txt" / "system_prompt_txt" / \
                            f"{topic_clean}_scenario{n_match.group(1)}_system_prompt.txt"
                if sp_file.exists():
                    try:
                        system_prompt = sp_file.read_text("utf-8").strip() or None
                    except Exception:
                        pass

                row, row_bytes = build_row(
                    dialogue_id, para_dim, topic,
                    scenario_id, scenario_desc,
                    turns, dialogue_dir / "individual",
                    system_prompt=system_prompt,
                )
                writer.add(row, row_bytes)
                log.info("Processed: %s", dialogue_id)

    for writer in writers.values():
        writer.close()
    log.info("All done. Parquet files written to: %s", out_dir)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Convert syn_data outputs → chunked Parquet files"
    )
    p.add_argument(
        "--data-dir", default="TEST_syn_data",
        help="Root data directory (default: TEST_syn_data)",
    )
    p.add_argument(
        "--out-dir", default="output_parquet",
        help="Output directory for Parquet parts (default: output_parquet)",
    )
    p.add_argument(
        "--chunk-gb", type=float, default=1.0,
        help="Approximate max GB per Parquet part (default: 1.0)",
    )
    p.add_argument(
        "--para-dims", nargs="+",
        default=["speed", "pitch", "gender", "emotion", "volume", "age", "multi"],
        help="Only convert these para_dim names (default: speed pitch gender emotion volume age)",
    )
    p.add_argument(
        "--prefix", type=str, default=None,
        help="Override output filename prefix (default: use para_dim name)",
    )
    args = p.parse_args()

    convert(
        data_dir    = args.data_dir,
        out_dir          = args.out_dir,
        chunk_bytes      = int(args.chunk_gb * 1024 ** 3),
        para_dims        = args.para_dims,
        prefix_override  = args.prefix,
    )

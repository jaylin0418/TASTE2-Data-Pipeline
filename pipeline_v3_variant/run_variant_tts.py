"""
Unified TTS pipeline for the variant dataset.
Supports IndexTTS-2 and BreezyVoice backends.

Usage:
  # IndexTTS-2 (single topic)
  python run_variant_tts.py --engine indextts2 \\
      --data-root /work/$USER/syn_variant --topic Art --device cuda:1

  # IndexTTS-2 (all topics)
  python run_variant_tts.py --engine indextts2 \\
      --data-root /work/$USER/syn_variant --device cuda:1

  # BreezyVoice (single topic)
  python run_variant_tts.py --engine breezyvoice \\
      --data-root /work/$USER/syn_variant --topic Art \\
      --config conf/base_variant_breezy.yaml

  # BreezyVoice (all topics — iterates topic dirs)
  python run_variant_tts.py --engine breezyvoice \\
      --data-root /work/$USER/syn_variant \\
      --config conf/base_variant_breezy.yaml

Speaker assignment:
  Each base scenario gets a (user, agent) speaker pair drawn with equal 25%
  probability from MM / FF / MF / FM gender combos.
  Assignments are cached in speaker_assignments.json per topic dir (IndexTTS-2 only).

Paralinguistic control per engine:
  neutral        IndexTTS-2: no control          BreezyVoice: neutral ref wav
  emotion_*      IndexTTS-2: emo_vector           BreezyVoice: emotion ref wav
  whisper        IndexTTS-2: emo_audio_prompt     BreezyVoice: whisper ref wav
  speed_slow/fast both: TTS then time_stretch(0.70/1.38)
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
# syn_para_breezy lives in pipeline_v2_para/
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "pipeline_v2_para"))

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import csv
import shutil
import subprocess
import tempfile

import librosa
import numpy as np
import torch
import torchaudio

# ── Shared constants ──────────────────────────────────────────────────────────
# Paths can be overridden via environment variables:
#   INDEXTTS_DIR         path to cog-IndexTTS-2 repo clone
#   REF_AUDIO_ROOT       path to ref audio dir (emotion wav files, organized by gender/)
#   WHISPER_EMO_REF      path to whisper_emo_ref.wav
#   BREEZYVOICE_REPO_DIR path to BreezyVoice repo clone
#   SYN_PYTHON           python binary for BreezyVoice subprocess (default: current interpreter)

INDEXTTS_DIR   = os.environ.get("INDEXTTS_DIR",
                     str(Path("/work") / os.environ.get("USER", "user") / "cog-IndexTTS-2"))
REF_AUDIO_ROOT = Path(os.environ.get("REF_AUDIO_ROOT",
                     str(Path.home() / "ref_audio" / "eleven_lab_emotion")))

MALE_SPEAKERS   = ["ranbir", "roger", "charlie", "george", "callum", "harry"]
FEMALE_SPEAKERS = ["river", "bella", "sarah", "laura"]
ALL_SPEAKERS    = MALE_SPEAKERS + FEMALE_SPEAKERS
SPEAKER_GENDER  = {s: "male" for s in MALE_SPEAKERS} | {s: "female" for s in FEMALE_SPEAKERS}

# IndexTTS-2 emo_vector order: happy, angry, sad, afraid, disgusted, melancholic, surprised, calm
EMOTION_VECTORS: Dict[str, List[float]] = {
    "angry":     [0.0, 1.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "sad":       [0.0, 0.0, 1.2, 0.0, 0.0, 0.6, 0.0, 0.0],
    "happy":     [1.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "surprised": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.2, 0.0],
    "fear":      [0.0, 0.0, 0.0, 1.2, 0.0, 0.0, 0.0, 0.0],
    "disgust":   [0.0, 0.0, 0.0, 0.0, 1.2, 0.0, 0.0, 0.0],
}

SPEED_FACTORS = {"speed_slow": 0.70, "speed_fast": 1.38}

# Whisper ref wav (shared across all speakers). Set WHISPER_EMO_REF to override.
WHISPER_EMO_REF = Path(os.environ.get("WHISPER_EMO_REF",
                     str(Path("/work") / os.environ.get("USER", "user") / "whisper_emo_ref.wav")))

# BreezyVoice (used by combined engine). Set BREEZYVOICE_REPO_DIR / SYN_PYTHON to override.
BREEZY_REPO = os.environ.get("BREEZYVOICE_REPO_DIR", str(Path.home() / "BreezyVoice"))
BREEZY_PY   = os.environ.get("SYN_PYTHON", sys.executable)

SAMPLE_RATE  = 24000
SILENCE_SEC  = 0.25


# ── Ref audio helpers ─────────────────────────────────────────────────────────

def _neutral_ref(speaker: str, n: int = 1) -> Path:
    gender = SPEAKER_GENDER[speaker]
    for i in (n, 1, 2, 3):
        p = REF_AUDIO_ROOT / gender / f"{speaker}_{i}.wav"
        if p.exists():
            return p
    raise FileNotFoundError(f"No neutral ref for {speaker}")




# ── Speaker selection (shared) ────────────────────────────────────────────────

def _pick_speakers() -> Tuple[str, str]:
    """Return (user_speaker, agent_speaker). Each gender combo is 25%."""
    combo = random.randint(0, 3)
    if combo == 0:
        user, agent = random.sample(MALE_SPEAKERS, 2)
    elif combo == 1:
        user, agent = random.sample(FEMALE_SPEAKERS, 2)
    elif combo == 2:
        user = random.choice(MALE_SPEAKERS)
        agent = random.choice(FEMALE_SPEAKERS)
    else:
        user = random.choice(FEMALE_SPEAKERS)
        agent = random.choice(MALE_SPEAKERS)
    return user, agent


def _load_or_create_assignments(topic_dir: Path) -> Dict[str, Dict]:
    path = topic_dir / "speaker_assignments.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _save_assignments(topic_dir: Path, assignments: Dict) -> None:
    path = topic_dir / "speaker_assignments.json"
    path.write_text(json.dumps(assignments, indent=2, ensure_ascii=False), encoding="utf-8")


def _get_speakers_for_scenario(assignments: Dict, base_name: str) -> Tuple[str, str]:
    if base_name in assignments:
        a = assignments[base_name]
        return a["user"], a["agent"]
    user, agent = _pick_speakers()
    assignments[base_name] = {"user": user, "agent": agent}
    return user, agent


# ── Dialogue parsing ──────────────────────────────────────────────────────────

_CTRL_TAG_RE = re.compile(r"^\([^)]+\)\s*")


def _strip_control_tag(text: str) -> str:
    return _CTRL_TAG_RE.sub("", text).strip()


def parse_dialogue(txt_path: Path) -> List[Tuple[str, str]]:
    turns = []
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("User: "):
            turns.append(("User", _strip_control_tag(line[6:].strip())))
        elif line.startswith("Agent: "):
            turns.append(("Agent", _strip_control_tag(line[7:].strip())))
    return turns


def _base_scenario_name(stem: str) -> str:
    for suffix in ("_emotion_angry", "_emotion_sad", "_emotion_happy",
                   "_emotion_surprised", "_emotion_fear", "_emotion_disgust",
                   "_speed_slow", "_speed_fast", "_whisper"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _variant_type_from_stem(stem: str) -> str:
    for suffix in ("emotion_angry", "emotion_sad", "emotion_happy",
                   "emotion_surprised", "emotion_fear", "emotion_disgust",
                   "speed_slow", "speed_fast", "whisper"):
        if stem.endswith(f"_{suffix}"):
            return suffix
    return "neutral"


# ── Audio utilities ───────────────────────────────────────────────────────────

def _time_stretch_wav(wav_path: Path, factor: float) -> None:
    wav, sr = torchaudio.load(str(wav_path))
    stretched = librosa.effects.time_stretch(wav.squeeze(0).numpy(), rate=factor)
    torchaudio.save(str(wav_path), torch.from_numpy(stretched).unsqueeze(0), sr)


def _build_stereo_full(
    turns: List[Tuple[str, Path]],
    out_path: Path,
    sample_rate: int = SAMPLE_RATE,
) -> None:
    silence = torch.zeros(1, int(sample_rate * SILENCE_SEC))
    l_segs: List[torch.Tensor] = []
    r_segs: List[torch.Tensor] = []

    for role, wav_path in turns:
        if not wav_path.exists():
            continue
        wav, sr = torchaudio.load(str(wav_path))
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != sample_rate:
            wav = torchaudio.functional.resample(wav, sr, sample_rate)
        if role == "User":
            l_segs.append(wav); r_segs.append(torch.zeros_like(wav))
        else:
            l_segs.append(torch.zeros_like(wav)); r_segs.append(wav)
        l_segs.append(silence.clone()); r_segs.append(silence.clone())

    if not l_segs:
        return
    l = torch.cat(l_segs, dim=1)
    r = torch.cat(r_segs, dim=1)
    n = min(l.shape[1], r.shape[1])
    torchaudio.save(str(out_path), torch.cat([l[:, :n], r[:, :n]], dim=0), sample_rate)


# ── IndexTTS-2 engine ─────────────────────────────────────────────────────────

def _tts_params_indextts2(
    role: str,
    variant_type: str,
    turn_idx: int,
    control_turn_idx: int,
    agent_speaker: str,
) -> Dict:
    result: Dict = {"emo_vector": None, "emo_audio_prompt": None, "post_stretch_factor": None}
    if role == "User" or turn_idx < control_turn_idx:
        return result

    emotion_key = variant_type[8:] if variant_type.startswith("emotion_") else variant_type
    if emotion_key in EMOTION_VECTORS:
        result["emo_vector"] = EMOTION_VECTORS[emotion_key]
    elif variant_type == "whisper":
        result["emo_audio_prompt"] = str(WHISPER_EMO_REF)
    elif variant_type in SPEED_FACTORS:
        result["post_stretch_factor"] = SPEED_FACTORS[variant_type]
    return result


def _para_tags(variant_type: str, role: str, turn_idx: int, control_turn_idx: int) -> Dict:
    if role == "User" or turn_idx < control_turn_idx:
        return {}
    if variant_type.startswith("emotion_"):
        return {"emotion": variant_type[8:]}
    if variant_type in ("speed_slow", "speed_fast"):
        return {"speed": variant_type[6:]}
    if variant_type == "whisper":
        return {"whisper": "true"}
    return {}


def synthesize_dialogue_indextts2(
    tts,
    txt_path: Path,
    para_json_path: Path,
    wav_out_dir: Path,
    user_speaker: str,
    agent_speaker: str,
    ref_num: int = 1,
) -> Optional[List[Dict]]:
    sidecar = json.loads(para_json_path.read_text(encoding="utf-8"))
    variant_type    = sidecar.get("variant_type", "neutral")
    controls        = sidecar.get("controls", [])
    control_turn_idx = controls[0]["turn_idx"] if controls else 9999

    turns = parse_dialogue(txt_path)
    if not turns:
        print(f"  [WARN] Empty dialogue: {txt_path.name}")
        return None

    individual_dir = wav_out_dir / "individual"
    individual_dir.mkdir(parents=True, exist_ok=True)

    user_ref  = str(_neutral_ref(user_speaker,  ref_num))
    agent_ref = str(_neutral_ref(agent_speaker, ref_num))

    turn_metadata: List[Dict] = []
    wav_turn_list: List[Tuple[str, Path]] = []

    for turn_idx, (role, text) in enumerate(turns):
        spk_name = user_speaker if role == "User" else agent_speaker
        spk_ref  = user_ref    if role == "User" else agent_ref
        params   = _tts_params_indextts2(role, variant_type, turn_idx, control_turn_idx, agent_speaker)

        fname    = f"turn_{turn_idx:03d}_{role}.wav"
        out_path = individual_dir / fname

        if out_path.exists():
            wav_turn_list.append((role, out_path))
            info = torchaudio.info(str(out_path))
            dur  = info.num_frames / info.sample_rate
        else:
            try:
                kwargs: Dict = {"spk_audio_prompt": spk_ref, "text": text, "output_path": str(out_path)}
                if params["emo_vector"] is not None:
                    kwargs["emo_vector"] = params["emo_vector"]
                if params["emo_audio_prompt"] is not None:
                    kwargs["emo_audio_prompt"] = params["emo_audio_prompt"]
                tts.infer(**kwargs)
                if params["post_stretch_factor"] is not None:
                    _time_stretch_wav(out_path, params["post_stretch_factor"])
            except Exception as e:
                print(f"  [ERROR] turn {turn_idx} of {txt_path.name}: {e}")
                return None
            info = torchaudio.info(str(out_path))
            dur  = info.num_frames / info.sample_rate
            wav_turn_list.append((role, out_path))

        turn_metadata.append({
            "turn_idx": turn_idx, "role": role,
            "speaker": role.lower(), "speaker_name": spk_name, "text": text,
            "para_tags": _para_tags(variant_type, role, turn_idx, control_turn_idx),
            "audio_path": str(out_path), "audio_duration": dur,
            "variant_type": variant_type,
        })

    (individual_dir / "turn_metadata.json").write_text(
        json.dumps(turn_metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _build_stereo_full(wav_turn_list, wav_out_dir / "full.wav")
    return turn_metadata


def _process_topic_indextts2(tts, topic_dir: Path, overwrite: bool = False, max_dialogues: int = 0, scenario: str = "") -> None:
    txt_dir = topic_dir / "txt" / "dialogue"
    wav_root = topic_dir / "wav"
    if not txt_dir.exists():
        print(f"  [SKIP] No txt/dialogue dir: {txt_dir}")
        return

    dialogues = [
        (p.stem, p, txt_dir / f"{p.stem}_para.json")
        for p in sorted(txt_dir.glob("*.txt"))
        if (txt_dir / f"{p.stem}_para.json").exists()
        and (not scenario or p.stem == scenario or p.stem.startswith(scenario + "_"))
    ]
    if not dialogues:
        print(f"  [SKIP] No .txt+para.json pairs in {txt_dir}")
        return
    if max_dialogues > 0:
        dialogues = dialogues[:max_dialogues]

    assignments     = _load_or_create_assignments(topic_dir)
    scenario_ref_num: Dict[str, int] = {}
    done = skipped = failed = 0

    for i, (stem, txt_path, para_json) in enumerate(dialogues, 1):
        base = _base_scenario_name(stem)
        user_spk, agent_spk = _get_speakers_for_scenario(assignments, base)
        if base not in scenario_ref_num:
            scenario_ref_num[base] = random.randint(1, 3)
        ref_num = scenario_ref_num[base]

        out_dir   = wav_root / stem
        meta_path = out_dir / "individual" / "turn_metadata.json"

        if not overwrite and meta_path.exists():
            skipped += 1
            if i % 50 == 0:
                print(f"  [{i}/{len(dialogues)}] {skipped} skipped, {done} done, {failed} failed")
            continue

        print(f"  [{i}/{len(dialogues)}] {stem}  (user={user_spk}, agent={agent_spk})", flush=True)
        t0 = time.perf_counter()
        result = synthesize_dialogue_indextts2(
            tts=tts, txt_path=txt_path, para_json_path=para_json,
            wav_out_dir=out_dir, user_speaker=user_spk, agent_speaker=agent_spk,
            ref_num=ref_num,
        )
        elapsed = time.perf_counter() - t0
        if result is not None:
            done += 1
            print(f"    -> OK ({elapsed:.1f}s, {len(result)} turns)", flush=True)
        else:
            failed += 1
            print(f"    -> FAILED ({elapsed:.1f}s)", flush=True)

    _save_assignments(topic_dir, assignments)
    print(f"\n  Topic done: {done} generated, {skipped} skipped, {failed} failed")


def _run_indextts2(args: argparse.Namespace) -> None:
    os.chdir(INDEXTTS_DIR)
    os.environ["HF_HUB_CACHE"] = str(Path(INDEXTTS_DIR) / "checkpoints" / "hf_cache")
    if INDEXTTS_DIR not in sys.path:
        sys.path.insert(0, INDEXTTS_DIR)
    from indextts import infer_v2  # noqa: E402

    variant_root = Path(args.data_root) / "variant"
    if not variant_root.exists():
        sys.exit(f"[ERROR] variant root not found: {variant_root}")

    topic_dirs = ([variant_root / args.topic] if args.topic
                  else sorted(p for p in variant_root.iterdir() if p.is_dir()))

    use_fp16 = args.device.startswith("cuda")
    print(f"Loading IndexTTS-2 on {args.device}...")
    tts = infer_v2.IndexTTS2(
        cfg_path=str(Path(INDEXTTS_DIR) / "checkpoints" / "config.yaml"),
        model_dir=str(Path(INDEXTTS_DIR) / "checkpoints"),
        use_fp16=use_fp16, device=args.device, use_cuda_kernel=use_fp16,
    )
    print("Model ready.\n")

    for topic_dir in topic_dirs:
        if not topic_dir.is_dir():
            print(f"[WARN] Topic dir not found: {topic_dir}")
            continue
        print(f"=== Topic: {topic_dir.name} ===")
        _process_topic_indextts2(tts, topic_dir, overwrite=args.overwrite,
                                 max_dialogues=args.max_dialogues, scenario=args.scenario)
        print()

    print("ALL DONE.")


# ── BreezyVoice engine ────────────────────────────────────────────────────────

def _run_breezyvoice(args: argparse.Namespace) -> None:
    from omegaconf import OmegaConf
    from syn_para_breezy import ParaPipeline, _prepare_para_paths

    config_path = Path(args.config).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    variant_root = Path(args.data_root) / "variant"
    if not variant_root.exists():
        sys.exit(f"[ERROR] variant root not found: {variant_root}")

    topic_dirs = ([variant_root / args.topic] if args.topic
                  else sorted(p for p in variant_root.iterdir() if p.is_dir()))

    for topic_dir in topic_dirs:
        if not topic_dir.is_dir():
            print(f"[WARN] Topic dir not found: {topic_dir}")
            continue
        topic = topic_dir.name
        print(f"=== Topic: {topic} ===")

        raw_cfg = OmegaConf.load(str(config_path))
        if "hydra" in raw_cfg:
            del raw_cfg["hydra"]
        raw_cfg["data_root"]         = str(Path(args.data_root))
        raw_cfg.scenario["topic"]    = topic
        raw_cfg["stages"]            = ["tts"]

        _prepare_para_paths(raw_cfg)
        ParaPipeline(raw_cfg).run()
        print()

    print("ALL DONE.")


# ── Combined engine ───────────────────────────────────────────────────────────
# neutral / speed_slow / speed_fast / User turns  → BreezyVoice
# emotion_* / whisper Agent turns (post-control)  → IndexTTS-2

# Variant types routed to IndexTTS-2 for Agent turns
_INDEXTTS_VARIANTS = {
    "emotion_angry", "emotion_sad", "emotion_happy",
    "emotion_surprised", "emotion_fear", "emotion_disgust",
    "whisper",
}


def _breezyvoice_batch(
    rows: List[Dict],          # [{ref_basename, text, out_stem}]
    ref_audio_map: Dict[str, Path],  # basename → actual wav path
    out_dir: Path,
) -> None:
    """Run BreezyVoice batch_inference.py for a list of turns."""
    if not rows:
        return

    tmp = Path(tempfile.mkdtemp(prefix="breezy_combined_"))
    try:
        # Copy ref audio files to temp dir
        for basename, src in ref_audio_map.items():
            dst = tmp / f"{basename}.wav"
            if not dst.exists():
                shutil.copy(str(src), str(dst))

        csv_path = tmp / "input.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "speaker_prompt_audio_filename",
                "content_to_synthesize",
                "output_audio_filename",
            ])
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    "speaker_prompt_audio_filename": row["ref_basename"],
                    "content_to_synthesize":          row["text"],
                    "output_audio_filename":           row["out_stem"],
                })

        subprocess.run(
            [BREEZY_PY, "batch_inference.py",
             "--csv_file",                    str(csv_path),
             "--speaker_prompt_audio_folder", str(tmp),
             "--output_audio_folder",         str(out_dir)],
            cwd=BREEZY_REPO, check=True,
            stdout=subprocess.DEVNULL,  # 抑制正常進度輸出
            # stderr 不抑制，讓錯誤訊息出現在 SLURM log
        )
    finally:
        shutil.rmtree(str(tmp), ignore_errors=True)


def _process_topic_combined(tts, topic_dir: Path, overwrite: bool = False, max_dialogues: int = 0, scenario: str = "") -> None:
    """
    Topic-level batching:
      Phase 1 — 收集所有 dialogue 的 BreezyVoice turns，一次呼叫 subprocess
      Phase 2 — 逐 dialogue 跑 IndexTTS-2 (emotion/whisper Agent turns)
      Phase 3 — 逐 dialogue 做 time_stretch、turn_metadata.json、full.wav
    """
    txt_dir  = topic_dir / "txt" / "dialogue"
    wav_root = topic_dir / "wav"
    if not txt_dir.exists():
        print(f"  [SKIP] No txt/dialogue dir: {txt_dir}")
        return

    dialogues = [
        (p.stem, p, txt_dir / f"{p.stem}_para.json")
        for p in sorted(txt_dir.glob("*.txt"))
        if (txt_dir / f"{p.stem}_para.json").exists()
        and (not scenario or p.stem == scenario or p.stem.startswith(scenario + "_"))
    ]
    if not dialogues:
        return
    if max_dialogues > 0:
        dialogues = dialogues[:max_dialogues]

    assignments      = _load_or_create_assignments(topic_dir)
    scenario_ref_num: Dict[str, int] = {}

    # ── Plan: parse all dialogues ─────────────────────────────────────────────
    # dialogue_plans[stem] = {turns, variant_type, control_turn_idx,
    #                          user_spk, agent_spk, ref_num, out_dir, skip}
    dialogue_plans: Dict[str, Dict] = {}

    for stem, txt_path, para_json in dialogues:
        base = _base_scenario_name(stem)
        user_spk, agent_spk = _get_speakers_for_scenario(assignments, base)
        if base not in scenario_ref_num:
            scenario_ref_num[base] = random.randint(1, 3)
        ref_num = scenario_ref_num[base]

        out_dir   = wav_root / stem
        meta_path = out_dir / "individual" / "turn_metadata.json"
        skip = not overwrite and meta_path.exists()

        sidecar          = json.loads(para_json.read_text(encoding="utf-8"))
        variant_type     = sidecar.get("variant_type", "neutral")
        controls         = sidecar.get("controls", [])
        control_turn_idx = controls[0]["turn_idx"] if controls else 9999
        turns            = parse_dialogue(txt_path)

        dialogue_plans[stem] = dict(
            turns=turns, variant_type=variant_type,
            control_turn_idx=control_turn_idx,
            user_spk=user_spk, agent_spk=agent_spk, ref_num=ref_num,
            out_dir=out_dir, skip=skip,
        )

    skipped = sum(1 for p in dialogue_plans.values() if p["skip"])
    todo    = {s: p for s, p in dialogue_plans.items() if not p["skip"]}
    if not todo:
        print(f"  全部 {skipped} 個已存在，略過。")
        _save_assignments(topic_dir, assignments)
        return

    # ── Phase 1: BreezyVoice batch (一次 subprocess for entire topic) ─────────
    print(f"  [Phase 1] BreezyVoice batch for {len(todo)} dialogues...", flush=True)

    # unique_stem 格式：{dialogue_stem}__turn_{idx:03d}_{role}
    breezy_rows:    List[Dict]      = []
    breezy_ref_map: Dict[str, Path] = {}
    # unique_stem → 目標路徑
    breezy_dest:    Dict[str, Path] = {}

    breezy_tmp_out = Path(tempfile.mkdtemp(prefix="breezy_topic_out_"))

    for stem, plan in todo.items():
        user_neutral  = _neutral_ref(plan["user_spk"],  plan["ref_num"])
        agent_neutral = _neutral_ref(plan["agent_spk"], plan["ref_num"])
        individual_dir = plan["out_dir"] / "individual"
        individual_dir.mkdir(parents=True, exist_ok=True)
        use_indextts = plan["variant_type"] in _INDEXTTS_VARIANTS

        for turn_idx, (role, text) in enumerate(plan["turns"]):
            out_path = individual_dir / f"turn_{turn_idx:03d}_{role}.wav"
            if out_path.exists():
                continue

            agent_needs_indextts = (
                role == "Agent" and use_indextts
                and turn_idx >= plan["control_turn_idx"]
            )
            if agent_needs_indextts:
                continue  # IndexTTS-2 handles this in Phase 2

            ref_path = user_neutral if role == "User" else agent_neutral
            basename = ref_path.stem
            breezy_ref_map[basename] = ref_path

            unique_stem = f"{stem}__turn_{turn_idx:03d}_{role}"
            breezy_rows.append({
                "ref_basename": basename,
                "text":         text,
                "out_stem":     unique_stem,
            })
            breezy_dest[unique_stem] = out_path

    try:
        if breezy_rows:
            _breezyvoice_batch(breezy_rows, breezy_ref_map, breezy_tmp_out)
            # 把輸出 wav 分配回各自的 individual/
            for unique_stem, dest_path in breezy_dest.items():
                src = breezy_tmp_out / f"{unique_stem}.wav"
                if src.exists():
                    shutil.move(str(src), str(dest_path))
                else:
                    print(f"  [WARN] BreezyVoice missing output: {unique_stem}", flush=True)
    except Exception as e:
        print(f"  [ERROR] BreezyVoice Phase 1 failed: {e}", flush=True)
        print(f"  IndexTTS-2 Phase 2 will still run for emotion/whisper turns.", flush=True)
    finally:
        shutil.rmtree(str(breezy_tmp_out), ignore_errors=True)
    print(f"  [Phase 1] 完成 ({len(breezy_rows)} turns)", flush=True)

    # ── Phase 2: IndexTTS-2 (emotion/whisper Agent turns) ─────────────────────
    print(f"  [Phase 2] IndexTTS-2 emotion/whisper turns...", flush=True)
    indextts_count = 0

    for stem, plan in todo.items():
        if plan["variant_type"] not in _INDEXTTS_VARIANTS:
            continue
        agent_neutral  = _neutral_ref(plan["agent_spk"], plan["ref_num"])
        individual_dir = plan["out_dir"] / "individual"

        for turn_idx, (role, text) in enumerate(plan["turns"]):
            if role != "Agent" or turn_idx < plan["control_turn_idx"]:
                continue
            out_path = individual_dir / f"turn_{turn_idx:03d}_{role}.wav"
            if out_path.exists():
                continue

            params = _tts_params_indextts2(
                role, plan["variant_type"], turn_idx,
                plan["control_turn_idx"], plan["agent_spk"],
            )
            kwargs: Dict = {
                "spk_audio_prompt": str(agent_neutral),
                "text":             text,
                "output_path":      str(out_path),
            }
            if params["emo_vector"] is not None:
                kwargs["emo_vector"] = params["emo_vector"]
            if params["emo_audio_prompt"] is not None:
                kwargs["emo_audio_prompt"] = params["emo_audio_prompt"]
            try:
                tts.infer(**kwargs)
                indextts_count += 1
            except Exception as e:
                print(f"  [ERROR] IndexTTS-2 {stem} turn {turn_idx}: {e}")

    print(f"  [Phase 2] 完成 ({indextts_count} turns)", flush=True)

    # ── Phase 3: time_stretch + turn_metadata.json + full.wav ─────────────────
    done = failed = 0
    for stem, plan in todo.items():
        individual_dir = plan["out_dir"] / "individual"

        # time_stretch
        if plan["variant_type"] in SPEED_FACTORS:
            factor = SPEED_FACTORS[plan["variant_type"]]
            for turn_idx, (role, _) in enumerate(plan["turns"]):
                if role != "Agent" or turn_idx < plan["control_turn_idx"]:
                    continue
                p = individual_dir / f"turn_{turn_idx:03d}_{role}.wav"
                if p.exists():
                    _time_stretch_wav(p, factor)

        # turn_metadata.json + full.wav
        turn_metadata: List[Dict] = []
        wav_turn_list: List[Tuple[str, Path]] = []
        ok = True
        for turn_idx, (role, text) in enumerate(plan["turns"]):
            out_path = individual_dir / f"turn_{turn_idx:03d}_{role}.wav"
            if not out_path.exists():
                ok = False
                continue
            info = torchaudio.info(str(out_path))
            dur  = info.num_frames / info.sample_rate
            spk_name = plan["user_spk"] if role == "User" else plan["agent_spk"]
            turn_metadata.append({
                "turn_idx": turn_idx, "role": role,
                "speaker": role.lower(), "speaker_name": spk_name, "text": text,
                "para_tags": _para_tags(
                    plan["variant_type"], role, turn_idx, plan["control_turn_idx"]
                ),
                "audio_path": str(out_path), "audio_duration": dur,
                "variant_type": plan["variant_type"],
            })
            wav_turn_list.append((role, out_path))

        if not ok:
            failed += 1
            print(f"  [WARN] {stem}: 部分 wav 缺失", flush=True)
            continue

        (individual_dir / "turn_metadata.json").write_text(
            json.dumps(turn_metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        _build_stereo_full(wav_turn_list, plan["out_dir"] / "full.wav")
        done += 1

    _save_assignments(topic_dir, assignments)
    print(f"\n  Topic done: {done} generated, {skipped} skipped, {failed} failed")


def _run_combined(args: argparse.Namespace) -> None:
    if not WHISPER_EMO_REF.exists():
        sys.exit(f"[ERROR] whisper emo ref not found: {WHISPER_EMO_REF}")

    os.chdir(INDEXTTS_DIR)
    if INDEXTTS_DIR not in sys.path:
        sys.path.insert(0, INDEXTTS_DIR)
    from indextts import infer_v2  # noqa: E402

    variant_root = Path(args.data_root) / "variant"
    if not variant_root.exists():
        sys.exit(f"[ERROR] variant root not found: {variant_root}")

    topic_dirs = ([variant_root / args.topic] if args.topic
                  else sorted(p for p in variant_root.iterdir() if p.is_dir()))

    use_fp16 = args.device.startswith("cuda")
    print(f"Loading IndexTTS-2 on {args.device}...")
    tts = infer_v2.IndexTTS2(
        cfg_path=str(Path(INDEXTTS_DIR) / "checkpoints" / "config.yaml"),
        model_dir=str(Path(INDEXTTS_DIR) / "checkpoints"),
        use_fp16=use_fp16, device=args.device, use_cuda_kernel=use_fp16,
    )
    print("Model ready.\n")

    for topic_dir in topic_dirs:
        if not topic_dir.is_dir():
            print(f"[WARN] Topic dir not found: {topic_dir}")
            continue
        print(f"=== Topic: {topic_dir.name} ===")
        _process_topic_combined(tts, topic_dir, overwrite=args.overwrite,
                                max_dialogues=args.max_dialogues, scenario=args.scenario)
        print()

    print("ALL DONE.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Variant dataset TTS (IndexTTS-2 or BreezyVoice)")
    parser.add_argument("--engine", choices=["indextts2", "breezyvoice", "combined"], required=True,
                        help="TTS backend to use")
    parser.add_argument("--data-root",
                        default=str(Path("/work") / os.environ.get("USER", "user") / "syn_variant"),
                        help="Root of the syn_variant dataset")
    parser.add_argument("--topic", default="",
                        help="Single topic to process (default: all)")
    parser.add_argument("--device", default="cuda:1",
                        help="Torch device, IndexTTS-2 only (default: cuda:1)")
    parser.add_argument("--config",
                        default=str(Path(__file__).parent.parent / "conf" / "base_variant_breezy.yaml"),
                        help="Config YAML path, BreezyVoice only")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-synthesize even if output exists (IndexTTS-2 only)")
    parser.add_argument("--max-dialogues", type=int, default=0,
                        help="Limit number of dialogues per topic (0 = no limit, for testing)")
    parser.add_argument("--scenario", type=str, default="",
                        help="Only process dialogues whose stem starts with this scenario name (e.g. Art_scenario1)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.engine == "indextts2":
        _run_indextts2(args)
    elif args.engine == "breezyvoice":
        _run_breezyvoice(args)
    else:
        _run_combined(args)


if __name__ == "__main__":
    main()

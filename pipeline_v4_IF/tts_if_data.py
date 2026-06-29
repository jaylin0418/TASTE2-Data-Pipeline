"""
TTS synthesis for English IF data using cog-IndexTTS-2.
Output format mirrors synthetic_dataset_TC_emo for TASTE training.

Structure:
  {output}/{category}/
    txt/
      dialogue/{category}_{idx:06d}.txt
      system_prompt_txt/{category}_{idx:06d}_system_prompt.txt
    wav/
      {category}_{idx:06d}/
        individual/
          turn00.wav          ← instruction (User)
          turn01.wav          ← response (Agent, styled)
          turn_metadata.json

Dialogue text format:
  User: {instruction}
  Agent: (style:{style}) {target_text}     [style=none → no tag]

System prompt (same for all):
  You are a helpful voice assistant. Follow the user's instructions carefully
  and respond in the exact speaking style they request.

Style → emotion vector (8-dim): happy, angry, sad, afraid, disgusted, melancholic, surprised, calm
  none/slow/fast → calm  [0,0,0,0,0,0,0,1.2]
  angry          →       [0,1.2,0,0,0,0,0,0]
  sad            →       [0,0,1.2,0,0,0,0,0]
  happy          →       [1.2,0,0,0,0,0,0,0]
  fearful        →       [0,0,0,1.2,0,0,0,0]
  disgusted      →       [0,0,0,0,1.2,0,0,0]
  surprised      →       [0,0,0,0,0,0,1.2,0]
  whisper        → emo_audio_prompt from emo_dir/{gender}/whisper_*.wav

Speed (slow/fast) → audiostretchy post-processing (ratio > 1 = slower, < 1 = faster).
"""

import argparse
import json
import os
import random
import sys
import tempfile
from pathlib import Path

import torch
import torchaudio

STYLE_TO_VECTOR = {
    "none":      [0, 0, 0, 0, 0, 0, 0, 1.2],
    "angry":     [0, 1.2, 0, 0, 0, 0, 0, 0],
    "sad":       [0, 0, 1.2, 0, 0, 0, 0, 0],
    "happy":     [1.2, 0, 0, 0, 0, 0, 0, 0],
    "fearful":   [0, 0, 0, 1.2, 0, 0, 0, 0],
    "disgusted": [0, 0, 0, 0, 1.2, 0, 0, 0],
    "surprised": [0, 0, 0, 0, 0, 0, 1.2, 0],
    "slow":      [0, 0, 0, 0, 0, 0, 0, 1.2],
    "fast":      [0, 0, 0, 0, 0, 0, 0, 1.2],
    "whisper":   None,
}

SPEED_RATIO = {"slow": 1.7, "fast": 0.55}

SYSTEM_PROMPT = (
    "You are a helpful voice assistant. "
    "Follow the user's instructions carefully and respond in the exact speaking style they request."
)

SAMPLE_RATE = 24000


# ── Helpers ───────────────────────────────────────────────────────────────────

def list_wavs(directory: str) -> list:
    p = Path(directory)
    if not p.exists():
        return []
    return sorted(str(f) for f in p.rglob("*") if f.suffix.lower() in (".wav", ".mp3"))




def apply_speed(wav: torch.Tensor, sr: int, ratio: float) -> torch.Tensor:
    try:
        from audiostretchy.stretch import AudioStretch
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            torchaudio.save(f.name, wav.cpu(), sr)
            in_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            out_path = f.name
        stretcher = AudioStretch()
        stretcher.open(in_path)
        stretcher.stretch(ratio=ratio)
        stretcher.save(out_path)
        wav_out, sr_out = torchaudio.load(out_path)
        os.remove(in_path)
        os.remove(out_path)
        if sr_out != sr:
            wav_out = torchaudio.functional.resample(wav_out, sr_out, sr)
        return wav_out
    except Exception as e:
        print(f"  [speed] audiostretchy failed ({e}), using resample fallback")
        new_sr = int(sr * ratio)
        wav2 = torchaudio.functional.resample(wav, sr, new_sr)
        return torchaudio.functional.resample(wav2, new_sr, sr)


def synthesize(tts, text: str, spk_ref: str,
               emo_vector=None, emo_audio_prompt=None, emo_alpha=1.0,
               speed_ratio=None) -> torch.Tensor:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
    try:
        tts.infer(
            spk_audio_prompt=spk_ref,
            text=text,
            output_path=tmp_path,
            emo_audio_prompt=emo_audio_prompt,
            emo_alpha=emo_alpha,
            emo_vector=emo_vector,
            use_emo_text=False,
            use_random=False,
            verbose=False,
        )
        wav, sr = torchaudio.load(tmp_path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        if speed_ratio is not None:
            wav = apply_speed(wav, SAMPLE_RATE, speed_ratio)
        return wav
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def get_spk_name(wav_path: str) -> str:
    return Path(wav_path).stem.rsplit("_", 1)[0]


# ── Output writers ─────────────────────────────────────────────────────────────

def write_dialogue(txt_dir: Path, sample_id: str, instruction: str, target_text: str, style: str):
    agent_text = f"(style:{style}) {target_text}" if style != "none" else target_text
    content = f"User: {instruction}\nAgent: {agent_text}\n"
    (txt_dir / "dialogue").mkdir(parents=True, exist_ok=True)
    (txt_dir / "dialogue" / f"{sample_id}.txt").write_text(content, encoding="utf-8")


def write_system_prompt(txt_dir: Path, sample_id: str):
    (txt_dir / "system_prompt_txt").mkdir(parents=True, exist_ok=True)
    (txt_dir / "system_prompt_txt" / f"{sample_id}_system_prompt.txt").write_text(
        SYSTEM_PROMPT, encoding="utf-8"
    )


def write_turn_metadata(ind_dir: Path, turns: list):
    (ind_dir / "turn_metadata.json").write_text(
        json.dumps(turns, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",         required=True)
    parser.add_argument("--output",        required=True)
    parser.add_argument("--model_dir",     required=True)
    parser.add_argument("--spk_dir",       required=True,
                        help="Speaker ref wavs (all speakers, male + female)")
    parser.add_argument("--whisper_ref",   required=True,
                        help="Single wav file used as emotion reference for whisper style")
    parser.add_argument("--sample_rate",   default=24000, type=int)
    parser.add_argument("--num_workers",   default=1, type=int,
                        help="Total number of parallel workers")
    parser.add_argument("--worker_id",     default=0, type=int,
                        help="This worker's ID (0-indexed). Processes examples where idx % num_workers == worker_id")
    parser.add_argument("--indextts_dir",  required=True,
                        help="Path to the cog-IndexTTS-2 repo root (added to sys.path for `from indextts import infer_v2`)")
    args = parser.parse_args()

    global SAMPLE_RATE
    SAMPLE_RATE = args.sample_rate

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    # Load examples
    all_examples = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_examples.append(json.loads(line))

    # Each worker handles its own slice (by global idx)
    examples = [(idx, ex) for idx, ex in enumerate(all_examples)
                if idx % args.num_workers == args.worker_id]
    print(f"Total examples: {len(all_examples)}  "
          f"This worker ({args.worker_id}/{args.num_workers}): {len(examples)}")

    # Speaker pool (all speakers, no gender restriction)
    all_spks = list_wavs(args.spk_dir)
    if len(all_spks) < 2:
        print(f"ERROR: need at least 2 wavs in --spk_dir {args.spk_dir}"); sys.exit(1)
    print(f"Speakers: {len(all_spks)}")

    # Load IndexTTS2
    print(f"Loading IndexTTS2 from {args.model_dir} ...")
    sys.path.insert(0, args.indextts_dir)
    from indextts import infer_v2

    # Patch QwenEmotion with a no-op fallback (we use emo_vector directly)
    original_qwen = getattr(infer_v2, "QwenEmotion", None)
    if original_qwen and not getattr(original_qwen, "_is_safe_wrapper", False):
        class _SafeQwenEmotion:
            _is_safe_wrapper = True
            def __init__(self, model_dir):
                try:
                    self._inner = original_qwen(model_dir)
                except Exception as e:
                    print(f">> QwenEmotion unavailable ({e}). Falling back to neutral.")
                    self._inner = None
            def inference(self, text):
                if self._inner is None:
                    return {"calm": 1.0}
                try:
                    return self._inner.inference(text)
                except Exception:
                    return {"calm": 1.0}
        infer_v2.QwenEmotion = _SafeQwenEmotion

    from indextts.infer_v2 import IndexTTS2
    tts = IndexTTS2(
        cfg_path=str(Path(args.model_dir) / "config.yaml"),
        model_dir=args.model_dir,
        use_fp16=torch.cuda.is_available(),
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        use_cuda_kernel=torch.cuda.is_available(),
    )
    print("Model loaded.\n")

    # Resume: collect all already-done sample_ids by checking existing turn_metadata.json
    done_ids: set = set()
    for meta_file in out_root.rglob("turn_metadata.json"):
        try:
            data = json.loads(meta_file.read_text(encoding="utf-8"))
            if data:
                sample_id = meta_file.parent.parent.name  # wav/{sample_id}/individual/
                done_ids.add(sample_id)
        except Exception:
            pass
    if done_ids:
        print(f"Resuming: {len(done_ids)} already done\n")

    total_done = 0
    total_err  = 0

    for idx, ex in examples:
        ability     = ex.get("ability", "unknown")
        style       = ex.get("style", "none")
        instruction = ex["instruction"]
        target_text = ex["target_text"]

        # Derive IF_type (folder = question type) and para_ability from ability string
        # acoustic_attributes/* are all read_aloud tasks with a style; folder = "read_aloud"
        # e.g. "acoustic_attributes/speed/slow"  → IF_type="read_aloud", para_ability="slow"
        #      "acoustic_attributes/emotion/angry"→ IF_type="read_aloud", para_ability="angry"
        #      "instruction_following/read_aloud" → IF_type="read_aloud", para_ability=None
        #      "instruction_following/counting"   → IF_type="counting",   para_ability=None
        parts = ability.split("/")
        if parts[0] == "acoustic_attributes":
            IF_type      = "read_aloud"
            para_ability = parts[2] if len(parts) >= 3 else None
        elif len(parts) >= 2:
            IF_type      = parts[1]
            para_ability = None
        else:
            IF_type      = ability
            para_ability = None

        sample_id = f"{IF_type}_{idx:06d}"
        if sample_id in done_ids:
            continue

        # Directories
        cat_dir  = out_root / IF_type
        txt_dir  = cat_dir / "txt"
        wav_root = cat_dir / "wav" / sample_id
        ind_dir  = wav_root / "individual"
        ind_dir.mkdir(parents=True, exist_ok=True)

        rng = random.Random(idx)
        user_spk  = rng.choice(all_spks)
        agent_spk = rng.choice([s for s in all_spks if s != user_spk])

        emo_vector  = STYLE_TO_VECTOR.get(style, STYLE_TO_VECTOR["none"])
        emo_audio   = None
        speed_ratio = SPEED_RATIO.get(style)

        if style == "whisper":
            emo_vector = None
            emo_audio  = args.whisper_ref

        try:
            # turn00: User instruction (neutral/calm)
            instr_wav = ind_dir / "turn00.wav"
            instr_audio = synthesize(
                tts, instruction, user_spk,
                emo_vector=[0, 0, 0, 0, 0, 0, 0, 1.2],
            )
            torchaudio.save(str(instr_wav), instr_audio, SAMPLE_RATE)

            # turn01: Agent response (styled)
            resp_wav = ind_dir / "turn01.wav"
            resp_audio = synthesize(
                tts, target_text, agent_spk,
                emo_vector=emo_vector,
                emo_audio_prompt=emo_audio,
                emo_alpha=1.0,
                speed_ratio=speed_ratio,
            )
            torchaudio.save(str(resp_wav), resp_audio, SAMPLE_RATE)

            instr_dur = instr_audio.shape[-1] / SAMPLE_RATE
            resp_dur  = resp_audio.shape[-1]  / SAMPLE_RATE
            cumulative = 0.0

            turns = []
            for turn_idx, (role, speaker, text, spk_ref, wav_path, dur) in enumerate([
                ("User",  "user",  instruction,  user_spk,  str(instr_wav), instr_dur),
                ("Agent", "agent", target_text,  agent_spk, str(resp_wav),  resp_dur),
            ]):
                turns.append({
                    "turn_idx":    turn_idx,
                    "role":        role,
                    "speaker":     speaker,
                    "text":        text,  # clean text only; style tag goes in dialogue .txt, not here
                    "IF_type":     IF_type,
                    "para_ability": para_ability,
                    "style":       style if role == "Agent" else None,
                    "ability":     ability,
                    "lang":        ex.get("lang", "en"),
                    "para_tags": {
                        "gender":  None,
                        "pitch":   None,
                        "speed":   "slow" if style == "slow" else ("fast" if style == "fast" else None),
                        "volume":  None,
                        "emotion": style if style not in ("none", "slow", "fast") else None,
                    },
                    "audio_path":    wav_path,
                    "audio_start":   round(cumulative, 6),
                    "audio_end":     round(cumulative + dur, 6),
                    "audio_duration": round(dur, 6),
                    "speaker_reference": spk_ref,
                    "speaker_reference_id": Path(spk_ref).name,
                })
                cumulative += dur

            write_turn_metadata(ind_dir, turns)
            write_dialogue(txt_dir, sample_id, instruction, target_text, style)
            write_system_prompt(txt_dir, sample_id)

            total_done += 1
            print(f"[{idx:05d}/{len(examples)}] {sample_id}  style={style:<10} "
                  f"{instr_dur:.1f}s+{resp_dur:.1f}s  {instruction[:50]}")

        except Exception as e:
            import traceback
            total_err += 1
            print(f"[{idx:05d}] ERROR {sample_id}: {e}")
            traceback.print_exc()

    print(f"\nDone: {total_done} new, {total_err} errors")
    print(f"Output: {out_root}")


if __name__ == "__main__":
    main()

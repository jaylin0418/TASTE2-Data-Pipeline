"""
Paralinguistic control data generation pipeline (BreezyVoice).

Each dialogue contains 2–4 paralinguistic control requests spread across turns
(speed / volume / pitch only — no emotion, since this dataset trains control).

Agent turns carry inline para tags showing cumulative active effects:
  Agent: (speed:fast, volume:loud) 好的，我說快一點大聲一點...

For each generated txt a mirror txt is also saved where speed/volume/pitch are
swapped (fast↔slow, loud↔quiet, high↔low) in both user phrases and agent tags.

Usage:
    python syn_para_breezy.py scenario.topic=Travel scenario.n=5
"""
from __future__ import annotations

import sys
import re
import json
import time
import random
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

# syn_ver2_breezy lives in dialogue_v1/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dialogue_v1"))

import torch
import torchaudio
import librosa
import numpy as np
from omegaconf import OmegaConf
import hydra

from syn_ver2_breezy import (
    _extract_topic_label,
    _canonicalize_scenario_id,
    _next_topic_scenario_index,
    strip_leading_emotion_tag,
    mix_segments_to_stereo_and_save_clean,
    _dialogue_file_stem,
    chat_completion,
    generate_scenarios,
    convert_nested_json_to_jsonl,
    tts_batch,
    export_to_huggingface,
    generate_system_prompts,
    _read_scenario_system_prompt,
)
from tqdm import tqdm

# ── Emotion Chinese name mapping ─────────────────────────────────────────────
EMOTION_ZH: Dict[str, str] = {
    "afraid": "害怕", "amusement": "有趣", "angry": "生氣", "anxiety": "焦慮",
    "calm": "平靜", "compassion": "同情", "contentment": "滿足", "cry": "哭泣",
    "disappointment": "失望", "disgusted": "厭惡", "envy": "嫉妒",
    "excitement": "興奮", "frustration": "挫折", "gratitude": "感激",
    "grief": "悲傷", "guilt": "內疚", "happy": "開心", "hope": "充滿希望",
    "melancholic": "憂鬱", "neutral": "自然", "pride": "驕傲", "relief": "放鬆",
    "sad": "難過", "shame": "羞愧", "surprised": "驚訝", "sarcastic": "諷刺",
    "hysteria": "歇斯底里",
}

# ── Symmetric mirror pairs ───────────────────────────────────────────────────
MIRROR_PAIRS: Dict[str, Dict[str, str]] = {
    "speed":  {"fast": "slow",  "slow": "fast",
               "very_fast": "very_slow", "very_slow": "very_fast"},
    "volume": {"loud": "quiet", "quiet": "loud",
               "very_loud": "very_quiet", "very_quiet": "very_loud"},
    "pitch":  {"high": "low",   "low": "high",
               "very_high": "very_low", "very_low": "very_high"},
}

# ── Post-processing helpers ───────────────────────────────────────────────────

def _time_stretch(wav: torch.Tensor, sr: int, factor: float) -> torch.Tensor:
    audio_np = wav.squeeze(0).numpy()
    stretched = librosa.effects.time_stretch(audio_np, rate=factor)
    return torch.from_numpy(stretched).unsqueeze(0)


def _volume_scale(wav: torch.Tensor, factor: float) -> torch.Tensor:
    return torch.clamp(wav * factor, -1.0, 1.0)


def _pitch_shift(wav: torch.Tensor, sr: int, semitones: int) -> torch.Tensor:
    audio_np = wav.squeeze(0).numpy()
    shifted = librosa.effects.pitch_shift(audio_np, sr=sr, n_steps=semitones)
    return torch.from_numpy(shifted).unsqueeze(0)


def _apply_single_effect(wav: torch.Tensor, sr: int, ctrl: Dict) -> torch.Tensor:
    fn = str(ctrl.get("fn", ""))
    if fn == "time_stretch":
        return _time_stretch(wav, sr, float(ctrl["factor"]))
    elif fn == "volume_scale":
        return _volume_scale(wav, float(ctrl["factor"]))
    elif fn == "pitch_shift":
        return _pitch_shift(wav, sr, int(ctrl["semitones"]))
    return wav  # ref_audio / unknown → no-op


# ── Control helpers ───────────────────────────────────────────────────────────

def _build_para_tag(active_state: Dict[str, str]) -> str:
    """Build inline tag prefix like '(speed:fast, volume:loud) ' or '' if empty."""
    if not active_state:
        return ""
    order = ["emotion", "pitch", "speed", "volume"]
    parts = [f"{dim}:{active_state[dim]}" for dim in order if dim in active_state]
    parts += [f"{dim}:{val}" for dim, val in sorted(active_state.items()) if dim not in order]
    return f"({', '.join(parts)}) "


def _extract_para_tags(text: str) -> Tuple[str, Dict[str, str]]:
    """Extract and remove a leading (key:value, ...) tag block from text."""
    m = re.match(r"^\(([^)]+)\)\s*", text)
    if not m:
        return text, {}
    clean = text[m.end():]
    tags: Dict[str, str] = {}
    for pair in m.group(1).split(","):
        if ":" in pair:
            k, v = pair.strip().split(":", 1)
            tags[k.strip()] = v.strip()
    return clean, tags


def _determine_control_mode(
    has_had_control: bool,
    freq_first: float,
    freq_subsequent: float,
    is_first_pair: bool = False,
    freq_catchup: float = 0.9,
) -> bool:
    if has_had_control:
        return random.random() < freq_subsequent
    if is_first_pair:
        return random.random() < freq_first
    return random.random() < freq_catchup


def _select_control_target(
    category_history: Dict[str, str],
    dims_cfg: Dict,
    emotion_pool: Optional[List[str]] = None,
    dim_weights: Optional[Dict[str, float]] = None,
    normal_reset_prob: Optional[Dict[str, float]] = None,
) -> Optional[Dict]:
    """Pick one control to request this turn (prefers dims not yet in history).

    emotion_pool: if provided, emotion dim is included in candidates.
    dim_weights: optional {dim: weight} for sampling bias.
    normal_reset_prob: per-dim probability of resetting to normal when dim is already active.
    Returns a dict with dim/value/control_key/fn + factor or semitones.
    """
    by_dim: Dict[str, List[str]] = {}
    for ck, dc in dims_cfg.items():
        d = str(dc.get("dim", ""))
        if d == "emotion" and not emotion_pool:
            continue
        by_dim.setdefault(d, []).append(ck)

    # Unused dims get full weight; used dims get 0.4x weight (can change or reset)
    all_dims = list(by_dim.keys())
    weights = []
    for d in all_dims:
        base = float((dim_weights or {}).get(d, 1.0))
        weights.append(base if d not in category_history else base * 0.4)
    if not all_dims:
        return None

    dim = random.choices(all_dims, weights=weights, k=1)[0]

    # Emotion: value is picked dynamically from pool (not fixed in yaml)
    if dim == "emotion" and emotion_pool:
        current_emo = category_history.get("emotion")
        pool = [e for e in emotion_pool if e != current_emo] or emotion_pool
        emotion = random.choice(pool)
        emo_zh = EMOTION_ZH.get(emotion, emotion)
        ck = by_dim["emotion"][0]
        return {
            "dim": "emotion",
            "value": emotion,
            "control_key": ck,
            "phrase": f"你可以用{emo_zh}的語氣說話嗎",
            "fn": "ref_audio",
        }

    current_value = category_history.get(dim)
    is_active = current_value is not None

    # If dim is active, check if we should reset to normal
    if is_active and normal_reset_prob:
        prob = float((normal_reset_prob or {}).get(dim, 0.0))
        normal_cks = [ck for ck in by_dim[dim] if str(dims_cfg[ck].get("value")) == "normal"]
        if normal_cks and random.random() < prob:
            ck = normal_cks[0]
            dc = dims_cfg[ck]
            phrases = list(dc.get("user_requests", []))
            return {
                "dim": dim,
                "value": "normal",
                "control_key": ck,
                "phrase": random.choice(phrases) if phrases else "",
                "fn": str(dc.get("fn", "time_stretch")),
                **({} if "factor" not in dc else {"factor": float(dc["factor"])}),
                **({} if "semitones" not in dc else {"semitones": int(dc["semitones"])}),
            }

    # Exclude current value and normal (normal handled above)
    ck_list = [
        ck for ck in by_dim[dim]
        if str(dims_cfg[ck].get("value")) != current_value
        and str(dims_cfg[ck].get("value")) != "normal"
    ] or by_dim[dim]
    ck = random.choice(ck_list)

    dc = dims_cfg[ck]
    phrases = list(dc.get("user_requests", []))
    entry: Dict[str, Any] = {
        "dim": dim,
        "value": dc.get("value"),
        "control_key": ck,
        "phrase": random.choice(phrases) if phrases else "",
        "fn": str(dc.get("fn", "ref_audio")),
    }
    if "factor" in dc and dc["factor"] is not None:
        entry["factor"] = float(dc["factor"])
    if "semitones" in dc and dc["semitones"] is not None:
        entry["semitones"] = int(dc["semitones"])
    return entry


def _mirror_control(ctrl: Dict, dims_cfg: Dict) -> Dict:
    """Return a mirrored copy of a single control dict."""
    m = dict(ctrl)
    dim, value = str(ctrl.get("dim", "")), str(ctrl.get("value", ""))

    # Emotion mirror → neutral (no emotional tag); remove the emotion request phrase
    if dim == "emotion":
        m.update(value="normal", phrase="")
        return m

    if dim not in MIRROR_PAIRS or value not in MIRROR_PAIRS[dim]:
        return m
    mirror_value = MIRROR_PAIRS[dim][value]
    for ck, dc in dims_cfg.items():
        if str(dc.get("dim", "")) == dim and str(dc.get("value", "")) == mirror_value:
            mirror_phrases = list(dc.get("user_requests", []))
            m.update(
                value=mirror_value,
                control_key=ck,
                phrase=random.choice(mirror_phrases) if mirror_phrases else ctrl.get("phrase", ""),
                fn=str(dc.get("fn", "ref_audio")),
            )
            m.pop("factor", None)
            m.pop("semitones", None)
            if "factor" in dc and dc["factor"] is not None:
                m["factor"] = float(dc["factor"])
            if "semitones" in dc and dc["semitones"] is not None:
                m["semitones"] = int(dc["semitones"])
            break
    return m


# ── Mirror txt generation ─────────────────────────────────────────────────────

def _build_mirror_txt(
    original_lines: List[str],
    original_controls: List[Dict],
    mirror_controls: List[Dict],
    single_control: bool = False,
) -> List[str]:
    """Build mirror dialogue lines.

    - User turns: replace injected control phrases with mirror phrases
    - Agent turns: rebuild sticky tags from mirror_controls category history
    """
    # User phrase replacements: turn_idx → (orig_phrase, mirror_phrase)
    user_replacements: Dict[int, Tuple[str, str]] = {}
    for orig, mirr in zip(original_controls, mirror_controls):
        op, mp = orig.get("phrase", ""), mirr.get("phrase", "")
        if op and op != mp:
            user_replacements[orig["turn_idx"]] = (op, mp)

    def _mirror_history_at(line_idx: int) -> Dict[str, str]:
        history: Dict[str, str] = {}
        prior = [c for c in mirror_controls if c.get("turn_idx", 9999) < line_idx]
        if single_control and prior:
            prior = [prior[-1]]
        for c in prior:
            if str(c.get("value", "")) == "normal":
                history.pop(c["dim"], None)
            else:
                history[c["dim"]] = str(c["value"])
        return history

    mirror_lines = []
    for i, line in enumerate(original_lines):
        if line.startswith("User: "):
            if i in user_replacements:
                op, mp = user_replacements[i]
                line = line.replace(op, mp, 1)
            mirror_lines.append(line)
        elif line.startswith("Agent: "):
            clean, _ = _extract_para_tags(line[len("Agent: "):])
            mirror_tag = _build_para_tag(_mirror_history_at(i))
            mirror_lines.append(f"Agent: {mirror_tag}{clean}")
        else:
            mirror_lines.append(line)
    return mirror_lines


# ── Dialogue generation ───────────────────────────────────────────────────────

def generate_dialogues_v2_para(cfg) -> None:
    """Generate multi-control dialogues and their mirrors.

    Design (following 學長's approach):
    - Control timing: probabilistic per turn pair (freq_first / freq_subsequent)
    - Effects: sticky — category_history accumulates; all subsequent agent turns carry ALL tags
    - Tag source: LLM outputs the tag, code verifies and corrects if missing
    - Mirror: agent tags swapped (fast↔slow etc.), user phrases swapped via template lookup
    """
    scen_path = Path(cfg.scenario["out_file"]).with_suffix(".jsonl")
    if not scen_path.exists():
        raise FileNotFoundError(f"Scenario JSONL not found: {scen_path}")

    out_dir = Path(cfg.dialogue["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = [json.loads(l) for l in scen_path.read_text(encoding="utf-8").splitlines()]

    dims_cfg = OmegaConf.to_container(cfg.para.dims, resolve=True)
    enabled_dims: Optional[List[str]] = list(cfg.para.get("enabled_dims") or []) or None
    if enabled_dims:
        dims_cfg = {ck: dc for ck, dc in dims_cfg.items() if str(dc.get("dim", "")) in enabled_dims}
    dim_weights: Dict[str, float] = OmegaConf.to_container(cfg.para.get("dim_weights", {}), resolve=True)
    allow_normal_reset: bool = bool(cfg.para.get("allow_normal_reset", True))
    single_control: bool = bool(cfg.para.get("single_control", False))
    if not allow_normal_reset:
        dims_cfg = {ck: dc for ck, dc in dims_cfg.items() if str(dc.get("value", "")) != "normal"}
    normal_reset_prob: Dict[str, float] = OmegaConf.to_container(cfg.para.get("normal_reset_prob", {}), resolve=True) if allow_normal_reset else {}
    agent_acks_cfg = OmegaConf.to_container(cfg.para.get("agent_ack", {}), resolve=True)  # unused (dead code)

    user_model = cfg.dialogue["user_model"]
    agent_model = cfg.dialogue["agent_model"]
    user_system_prompt = str(cfg.dialogue["user_prompt"])
    agent_system_prompt = str(cfg.dialogue["agent_prompt"])
    max_turns = cfg.dialogue.get("max_turns", 14)
    min_turns = cfg.dialogue.get("min_turns", 10)
    dialogues_per_scenario = cfg.dialogue.get("per_scenario", 1)
    max_retries = cfg.dialogue.get("max_retries", 3)
    user_gen_kwargs = dict(cfg.dialogue.get("user_gen", {})) if cfg.dialogue.get("user_gen") else {}
    agent_gen_kwargs = dict(cfg.dialogue.get("agent_gen", {})) if cfg.dialogue.get("agent_gen") else {}
    topic_label = _extract_topic_label(str(cfg.scenario.get("topic", "unknown")))
    emotion_pool: List[str] = list(cfg.get("emotion", [])) if (not enabled_dims or "emotion" in enabled_dims) else []

    freq_first = float(cfg.para.get("control_request_frequency_first", 0.5))
    freq_catchup = float(cfg.para.get("control_request_frequency_catchup", 0.9))
    freq_subsequent = float(cfg.para.get("control_request_frequency_subsequent", 0.0))

    data_root = Path(cfg.get("data_root", "TEST_syn_para"))
    mode_name = str(cfg.get("mode_name", "para"))

    for scenario in tqdm(scenarios, desc="para dialogues"):
        scenario_id = _canonicalize_scenario_id(scenario.get("id", ""))
        scenario_description = str(scenario.get("description", "")).strip()
        context_info = (
            f"Topic（分類名稱，不要直接講出英文）：{topic_label}\n"
            f"Scenario ID：{scenario_id}\n"
            f"Scenario description：{scenario_description}\n\n"
        )

        # Per-scenario system prompt (generated by system_prompt stage); fall back to generic
        scenario_sys_prompt = _read_scenario_system_prompt(
            data_root=data_root,
            mode_name=mode_name,
            topic_label=topic_label,
            scenario_id=scenario_id,
            cfg=cfg,
        )
        effective_agent_system_prompt = (scenario_sys_prompt.strip() if scenario_sys_prompt else "") or agent_system_prompt

        for dialogue_idx in range(dialogues_per_scenario):
            dialogue_stem = _dialogue_file_stem(
                scenario_id, dialogue_idx + 1, dialogues_per_scenario, topic_label=topic_label
            )
            out_path = out_dir / f"{dialogue_stem}.txt"
            mirror_path = out_dir / f"{dialogue_stem}_mirror.txt"
            if out_path.exists() and mirror_path.exists():
                logging.info("Skipping %s (exists)", dialogue_stem)
                continue

            num_turns = random.randint(min_turns, max_turns)
            if num_turns % 2 != 0:
                num_turns += 1  # ensure even (end on agent turn)
            num_pairs = num_turns // 2

            dialogue_turns: List[str] = []
            conversation_history: List[str] = []
            category_history: Dict[str, str] = {}  # sticky accumulated para state
            has_had_control = False
            injected_controls: List[Dict] = []

            logging.info("Generating %s (%d turns)", dialogue_stem, num_turns)

            for pair_idx in range(num_pairs):
                is_first_pair = (pair_idx == 0)

                # ── Determine control mode for this turn pair ──────────────
                is_control = _determine_control_mode(
                    has_had_control, freq_first, freq_subsequent,
                    is_first_pair=is_first_pair, freq_catchup=freq_catchup,
                )
                ctrl = None
                if is_control:
                    ctrl = _select_control_target(category_history, dims_cfg, emotion_pool, dim_weights=dim_weights, normal_reset_prob=normal_reset_prob)
                    if ctrl:
                        has_had_control = True

                # ── User turn ──────────────────────────────────────────────
                user_turn_idx = len(dialogue_turns)
                if is_first_pair:
                    user_instruction = (
                        f"{context_info}"
                        f"現在是對話開始。請根據情境，產生使用者的第一句話。"
                    )
                else:
                    history_text = "\n".join(conversation_history)
                    user_instruction = (
                        f"{context_info}"
                        f"目前對話：\n{history_text}\n\n"
                        f"請根據代理人上一句回覆，產生使用者的下一句回應。"
                    )

                if ctrl and ctrl.get("phrase"):
                    user_instruction += (
                        f"\n\n【這一輪請自然地說出：「{ctrl['phrase']}」，"
                        f"然後繼續對話主題。整體輸出一行，不要換行。】"
                    )

                user_utterance = _llm_call(
                    user_model,
                    [{"role": "system", "content": user_system_prompt},
                     {"role": "user", "content": user_instruction}],
                    user_gen_kwargs, max_retries,
                )
                if user_utterance:
                    user_utterance = re.sub(r"^User:\s*", "", user_utterance, flags=re.IGNORECASE)
                    user_utterance = strip_leading_emotion_tag(user_utterance)
                if not user_utterance:
                    break

                if ctrl:
                    injected_controls.append({**ctrl, "turn_idx": user_turn_idx})
                    if str(ctrl["value"]) == "normal":
                        category_history.pop(ctrl["dim"], None)
                    else:
                        if single_control:
                            category_history.clear()
                        category_history[ctrl["dim"]] = str(ctrl["value"])

                dialogue_turns.append(f"User: {user_utterance}")
                conversation_history.append(f"User: {user_utterance}")

                # ── Agent turn ─────────────────────────────────────────────
                history_text = "\n".join(conversation_history)
                agent_instruction = (
                    f"{context_info}"
                    f"目前對話：\n{history_text}\n\n"
                    f"請回覆使用者上一句話：務實、可執行、自然口語。"
                    f"只輸出代理人要說的內容。"
                )

                if category_history:
                    expected_tag = _build_para_tag(category_history).strip()
                    if ctrl:
                        if ctrl["dim"] == "emotion":
                            emo_zh = EMOTION_ZH.get(str(ctrl["value"]), str(ctrl["value"]))
                            agent_instruction += (
                                f"\n\n【使用者請你用「{emo_zh}」的語氣說話，簡短自然地確認後繼續回答。"
                                f"整個回覆只能有一個標籤，放在最前面：{expected_tag} 你的回覆。一行輸出。】"
                            )
                        else:
                            agent_instruction += (
                                f"\n\n【使用者剛才請你調整說話方式，簡短自然地確認後繼續回答。"
                                f"整個回覆只能有一個標籤，放在最前面：{expected_tag} 你的回覆。一行輸出。】"
                            )
                    else:
                        agent_instruction += (
                            f"\n\n【請維持之前設定的說話風格。"
                            f"整個回覆只能有一個標籤，放在最前面：{expected_tag} 你的回覆。一行輸出。】"
                        )

                agent_utterance = _llm_call(
                    agent_model,
                    [{"role": "system", "content": effective_agent_system_prompt},
                     {"role": "user", "content": agent_instruction}],
                    agent_gen_kwargs, max_retries,
                )
                if agent_utterance:
                    agent_utterance = re.sub(r"^Agent:\s*", "", agent_utterance, flags=re.IGNORECASE)
                    # Let LLM's tag stand; if missing, add programmatically
                    if category_history and not agent_utterance.startswith("("):
                        agent_utterance = f"{_build_para_tag(category_history).strip()} {agent_utterance}"
                    # Remove any duplicate tag blocks that appear after the first one
                    agent_utterance = re.sub(r"(\([^)]+\))\s*\([^)]+\)", r"\1", agent_utterance)
                if not agent_utterance:
                    break

                dialogue_turns.append(f"Agent: {agent_utterance}")
                # History gets clean text (no tag) to avoid confusing the LLM
                clean_agent, _ = _extract_para_tags(agent_utterance)
                conversation_history.append(f"Agent: {clean_agent}")

            if not dialogue_turns:
                logging.warning("Empty dialogue for %s, skipping", dialogue_stem)
                continue

            # Save original txt + sidecar (speaker slot left empty; TTS fills it in)
            out_path.write_text("\n".join(dialogue_turns), encoding="utf-8")
            (out_dir / f"{dialogue_stem}_para.json").write_text(
                json.dumps({"controls": injected_controls, "speakers": {}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            # Build and save mirror (agent tags swapped, user phrases swapped via template)
            mirror_controls = [_mirror_control(c, dims_cfg) for c in injected_controls]
            mirror_lines = _build_mirror_txt(dialogue_turns, injected_controls, mirror_controls, single_control=single_control)
            mirror_path.write_text("\n".join(mirror_lines), encoding="utf-8")
            (out_dir / f"{dialogue_stem}_mirror_para.json").write_text(
                json.dumps({"controls": mirror_controls, "speakers": {}, "mirror_of": dialogue_stem}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            logging.info(
                "Saved %s + mirror (controls: %s)",
                dialogue_stem,
                [(c["dim"], c["value"]) for c in injected_controls],
            )


def _llm_call(model, messages, gen_kwargs, max_retries):
    for retry in range(max_retries):
        try:
            resp = chat_completion(model, messages, **gen_kwargs).strip()
            if resp and _is_valid_utterance(resp):
                return resp
            if resp:
                logging.warning("LLM output failed validation (retry %d): %r", retry + 1, resp[:80])
        except Exception as e:
            logging.error("LLM error retry %d/%d: %s", retry + 1, max_retries, e)
        if retry < max_retries - 1:
            time.sleep(2 ** retry)
    return None


def _is_valid_utterance(text: str) -> bool:
    """Basic sanity check: must contain Chinese characters and not be mostly garbage."""
    chinese_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    if chinese_chars < 3:
        return False
    # Reject if backslash density is high (common garbage pattern)
    if text.count("\\") > len(text) * 0.1:
        return False
    return True


# ── Post-processing pass ──────────────────────────────────────────────────────

def _postprocess_single_dialogue(
    dialogue_folder: Path,
    txt_dir: Path,
    sample_rate: int = 24000,
) -> None:
    """Apply para effects to a single dialogue folder immediately after TTS."""
    metadata_path = dialogue_folder / "individual" / "turn_metadata.json"
    if not metadata_path.exists():
        logging.warning("No turn_metadata.json in %s, skipping", dialogue_folder)
        return

    sidecar_path = txt_dir / f"{dialogue_folder.name}_para.json"
    if not sidecar_path.exists():
        m = re.match(r"^(.+?)_(?:para|[a-z]+)_(\d+)(_mirror)?$", dialogue_folder.name)
        if m:
            topic_part, idx, mirror_sfx = m.group(1), m.group(2), m.group(3) or ""
            alt = txt_dir / f"{topic_part}_scenario{idx}{mirror_sfx}_para.json"
            if alt.exists():
                sidecar_path = alt
    if not sidecar_path.exists():
        logging.warning("No sidecar for %s, skipping postprocess", dialogue_folder.name)
        return

    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    controls: List[Dict] = sidecar.get("controls", [])
    pp_controls = [c for c in controls if c.get("fn", "ref_audio") != "ref_audio"]
    if not pp_controls:
        return

    with open(metadata_path, encoding="utf-8") as f:
        metadata: List[Dict[str, Any]] = json.load(f)

    modified = False
    for entry in metadata:
        if str(entry.get("speaker", "")).lower() != "agent":
            continue
        turn_idx = int(entry.get("turn_idx", 0))
        active = [c for c in pp_controls if c.get("turn_idx", 9999) < turn_idx]
        if not active:
            continue
        turn_wav_path = Path(str(entry.get("audio_path", "")))
        if not turn_wav_path.exists():
            logging.warning("Missing wav: %s", turn_wav_path)
            continue
        wav, sr = torchaudio.load(str(turn_wav_path))
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != sample_rate:
            wav = torchaudio.functional.resample(wav, sr, sample_rate)
            sr = sample_rate
        for eff in sorted(active, key=lambda c: ["volume_scale", "pitch_shift", "time_stretch"].index(c.get("fn", "time_stretch")) if c.get("fn") in ["volume_scale", "pitch_shift", "time_stretch"] else 99):
            wav = _apply_single_effect(wav, sr, eff)
        torchaudio.save(str(turn_wav_path), wav, sr)
        entry["audio_duration"] = float(wav.shape[1]) / sr
        para_tags = dict(entry.get("para_tags") or {})
        for eff in active:
            para_tags[eff["dim"]] = eff["value"]
        entry["para_tags"] = para_tags
        modified = True

    if not modified:
        return

    full_path = dialogue_folder / "full.wav"
    _rebuild_full_wav(metadata, full_path, sample_rate)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logging.info("Post-processed: %s", dialogue_folder.name)


def apply_para_postprocess_cumulative(
    wav_dir: Path,
    txt_dir: Path,
    sample_rate: int = 24000,
) -> None:
    """Apply para effects to all dialogue folders in wav_dir."""
    dialogue_folders = sorted(p for p in wav_dir.iterdir() if p.is_dir())
    logging.info("Post-processing %d dialogue folders in %s", len(dialogue_folders), wav_dir)

    for dialogue_folder in tqdm(dialogue_folders, desc="postprocess"):
        _postprocess_single_dialogue(dialogue_folder, txt_dir, sample_rate)


def _rebuild_full_wav(metadata, output_path, sample_rate):
    segments: List[Tuple[str, torch.Tensor]] = []
    for entry in metadata:
        p = Path(str(entry.get("audio_path", "")))
        if not p.exists():
            continue
        wav, sr = torchaudio.load(str(p))
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != sample_rate:
            wav = torchaudio.functional.resample(wav, sr, sample_rate)
        role = "User" if str(entry.get("speaker", "")).lower() == "user" else "Agent"
        segments.append((role, wav))
    if segments:
        mix_segments_to_stereo_and_save_clean(
            audio_segments=segments, output=output_path, sample_rate=sample_rate
        )


# ── Variant mode: one neutral base → N controlled variants ───────────────────

def _build_variant_txt(
    base_lines: List[str],
    dim: str,
    value: str,
    phrase: str,
    injection_pair_idx: int,
) -> List[str]:
    """Derive a controlled variant from a neutral base dialogue.

    Inserts `phrase` into the user turn at `injection_pair_idx`, then adds
    `(dim:value)` to every subsequent agent turn.
    """
    tag = f"({dim}:{value})"
    variant_lines = []
    user_count = 0
    control_active = False

    for line in base_lines:
        if line.startswith("User: "):
            text = line[len("User: "):]
            if user_count == injection_pair_idx and phrase:
                text = f"{phrase}，{text}"
            user_count += 1
            if user_count > injection_pair_idx:
                control_active = True
            variant_lines.append(f"User: {text}")
        elif line.startswith("Agent: "):
            text = line[len("Agent: "):]
            if control_active:
                variant_lines.append(f"Agent: {tag} {text}")
            else:
                variant_lines.append(line)
        else:
            variant_lines.append(line)

    return variant_lines


def generate_dialogues_v2_variant(cfg) -> None:
    """Generate 1 neutral base + N controlled variants per scenario.

    LLM runs once to produce a clean neutral dialogue; `_build_variant_txt`
    derives each controlled variant without additional LLM calls.
    """
    scen_path = Path(cfg.scenario["out_file"]).with_suffix(".jsonl")
    if not scen_path.exists():
        raise FileNotFoundError(f"Scenario JSONL not found: {scen_path}")

    out_dir = Path(cfg.dialogue["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = [json.loads(l) for l in scen_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    variants_cfg: Dict[str, Any] = OmegaConf.to_container(cfg.variants, resolve=True)
    injection_pair_choices: List[int] = list(cfg.get("injection_pair_choices", [0, 1]))

    user_model = cfg.dialogue["user_model"]
    agent_model = cfg.dialogue["agent_model"]
    user_system_prompt = str(cfg.dialogue["user_prompt"])
    agent_system_prompt = str(cfg.dialogue["agent_prompt"])
    max_turns = cfg.dialogue.get("max_turns", 8)
    min_turns = cfg.dialogue.get("min_turns", 6)
    max_retries = cfg.dialogue.get("max_retries", 3)
    user_gen_kwargs = dict(cfg.dialogue.get("user_gen", {})) if cfg.dialogue.get("user_gen") else {}
    agent_gen_kwargs = dict(cfg.dialogue.get("agent_gen", {})) if cfg.dialogue.get("agent_gen") else {}
    topic_label = _extract_topic_label(str(cfg.scenario.get("topic", "unknown")))
    data_root = Path(cfg.get("data_root", "TEST_syn_variant"))
    mode_name = str(cfg.get("mode_name", "variant"))

    variant_keys = list(variants_cfg.keys())

    for scenario in tqdm(scenarios, desc="variant dialogues"):
        scenario_id = _canonicalize_scenario_id(scenario.get("id", ""))
        scenario_description = str(scenario.get("description", "")).strip()
        context_info = (
            f"Topic（分類名稱，不要直接講出英文）：{topic_label}\n"
            f"Scenario ID：{scenario_id}\n"
            f"Scenario description：{scenario_description}\n\n"
        )

        dialogue_stem = _dialogue_file_stem(scenario_id, 1, 1, topic_label=topic_label)
        neutral_path = out_dir / f"{dialogue_stem}.txt"
        all_variant_paths = [out_dir / f"{dialogue_stem}_{vk}.txt" for vk in variant_keys]

        if neutral_path.exists() and all(p.exists() for p in all_variant_paths):
            logging.info("Skipping %s (all variants exist)", dialogue_stem)
            continue

        # Per-scenario system prompt (falls back to generic agent_prompt)
        scenario_sys_prompt = _read_scenario_system_prompt(
            data_root=data_root,
            mode_name=mode_name,
            topic_label=topic_label,
            scenario_id=scenario_id,
            cfg=cfg,
        )
        effective_agent_sys = (scenario_sys_prompt.strip() if scenario_sys_prompt else "") or agent_system_prompt

        # ── Generate neutral base dialogue ────────────────────────────────────
        num_turns = random.randint(min_turns, max_turns)
        if num_turns % 2 != 0:
            num_turns += 1
        num_pairs = num_turns // 2

        dialogue_turns: List[str] = []
        conversation_history: List[str] = []

        logging.info("Generating base dialogue %s (%d turns)", dialogue_stem, num_turns)

        for pair_idx in range(num_pairs):
            if pair_idx == 0:
                user_instruction = (
                    f"{context_info}"
                    f"現在是對話開始。請根據情境，產生使用者的第一句話。"
                )
            else:
                history_text = "\n".join(conversation_history)
                user_instruction = (
                    f"{context_info}"
                    f"目前對話：\n{history_text}\n\n"
                    f"請根據代理人上一句回覆，產生使用者的下一句回應。"
                )

            user_utterance = _llm_call(
                user_model,
                [{"role": "system", "content": user_system_prompt},
                 {"role": "user", "content": user_instruction}],
                user_gen_kwargs, max_retries,
            )
            if user_utterance:
                user_utterance = re.sub(r"^User:\s*", "", user_utterance, flags=re.IGNORECASE)
                user_utterance = strip_leading_emotion_tag(user_utterance)
            if not user_utterance:
                break

            dialogue_turns.append(f"User: {user_utterance}")
            conversation_history.append(f"User: {user_utterance}")

            history_text = "\n".join(conversation_history)
            agent_instruction = (
                f"{context_info}"
                f"目前對話：\n{history_text}\n\n"
                f"請回覆使用者上一句話：務實、可執行、自然口語。只輸出代理人要說的內容。"
            )

            agent_utterance = _llm_call(
                agent_model,
                [{"role": "system", "content": effective_agent_sys},
                 {"role": "user", "content": agent_instruction}],
                agent_gen_kwargs, max_retries,
            )
            if agent_utterance:
                agent_utterance = re.sub(r"^Agent:\s*", "", agent_utterance, flags=re.IGNORECASE)
                agent_utterance = strip_leading_emotion_tag(agent_utterance)
            if not agent_utterance:
                break

            dialogue_turns.append(f"Agent: {agent_utterance}")
            conversation_history.append(f"Agent: {agent_utterance}")

        if len(dialogue_turns) < 4:
            logging.warning("Base dialogue too short for %s, skipping", dialogue_stem)
            continue

        # ── Save neutral ──────────────────────────────────────────────────────
        if not neutral_path.exists():
            neutral_path.write_text("\n".join(dialogue_turns), encoding="utf-8")
        neutral_para_path = out_dir / f"{dialogue_stem}_para.json"
        if not neutral_para_path.exists():
            neutral_para_path.write_text(
                json.dumps({"controls": [], "speakers": {}, "variant_type": "neutral"}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        # ── Build and save each controlled variant ────────────────────────────
        for variant_key, vcfg in variants_cfg.items():
            variant_path = out_dir / f"{dialogue_stem}_{variant_key}.txt"
            variant_para_path = out_dir / f"{dialogue_stem}_{variant_key}_para.json"
            if variant_path.exists() and variant_para_path.exists():
                continue

            dim = str(vcfg.get("dim", variant_key))
            value = str(vcfg.get("value", variant_key))
            user_requests = list(vcfg.get("user_requests", []))
            phrase = random.choice(user_requests) if user_requests else ""
            injection_pair_idx = random.choice(injection_pair_choices)

            variant_lines = _build_variant_txt(dialogue_turns, dim, value, phrase, injection_pair_idx)
            variant_path.write_text("\n".join(variant_lines), encoding="utf-8")

            # turn_idx = line index of the user turn where phrase was injected
            turn_idx = injection_pair_idx * 2
            ctrl_record: Dict[str, Any] = {
                "dim": dim,
                "value": value,
                "control_key": variant_key,
                "phrase": phrase,
                "fn": str(vcfg.get("fn", "ref_audio")),
                "turn_idx": turn_idx,
                "injection_pair_idx": injection_pair_idx,
            }
            if "factor" in vcfg:
                ctrl_record["factor"] = vcfg["factor"]
            if "semitones" in vcfg:
                ctrl_record["semitones"] = vcfg["semitones"]

            variant_para_path.write_text(
                json.dumps({"controls": [ctrl_record], "speakers": {}, "variant_type": variant_key},
                           ensure_ascii=False, indent=2),
                encoding="utf-8",
            )


# ── Pipeline orchestrator ─────────────────────────────────────────────────────

class ParaPipeline:
    def __init__(self, cfg):
        self.cfg = cfg

    def run(self):
        cfg = self.cfg
        stages = set(cfg.stages)

        logging.info(
            "ParaPipeline: topic=%s, mode=%s",
            cfg.scenario.get("topic"),
            cfg.get("mode_name"),
        )

        if "scenario" in stages:
            print("Generating scenarios...")
            generate_scenarios(cfg)
            original_path = Path(cfg.scenario["out_file"])
            scenario_txt_dir = original_path.parent.parent / "scenario_txt"
            topic_label = _extract_topic_label(str(cfg.scenario.get("topic", "unknown")))
            start_index = _next_topic_scenario_index(cfg, topic_label)
            convert_nested_json_to_jsonl(
                original_path,
                original_path.with_suffix(".jsonl"),
                scenario_txt_dir=scenario_txt_dir,
                topic_label=topic_label,
                start_index=start_index,
                cfg=cfg,
            )

        if "system_prompt" in stages:
            print("Generating scenario-level system prompts...")
            generate_system_prompts(cfg)

        if "dialogue" in stages:
            if cfg.get("variants_mode", False):
                print("Generating variant dialogues (neutral base + N controlled variants)...")
                generate_dialogues_v2_variant(cfg)
            else:
                print("Generating para dialogues (multi-control + mirrors)...")
                generate_dialogues_v2_para(cfg)

        if "tts" in stages:
            print("TTS (BreezyVoice) with per-dialogue postprocessing...")
            data_root = Path(cfg.get("data_root", "data"))
            mode_name = str(cfg.get("mode_name", "para"))
            topic = _extract_topic_label(str(cfg.scenario.get("topic", "unknown")))
            suffix = str(cfg.get("topic_folder_suffix", "") or "")
            txt_dir = data_root / mode_name / (topic + suffix) / "txt" / "dialogue"
            sample_rate = int(cfg.tts.get("sample_rate", 24000))

            def _on_dialogue_done(dialogue_folder: Path) -> None:
                _postprocess_single_dialogue(dialogue_folder, txt_dir, sample_rate)

            tts_batch(cfg, on_dialogue_done=_on_dialogue_done)

        if "huggingface" in stages:
            print("Exporting to HuggingFace...")
            export_to_huggingface(cfg)


# ── CLI entry ─────────────────────────────────────────────────────────────────

def _prepare_para_paths(cfg) -> None:
    """Resolve data_root to absolute and wire all stage paths under it.

    Unlike prepare_run_output_root, does NOT create a timestamped run_root dir.
    """
    data_root = Path(str(cfg.get("data_root", "TEST_syn_para"))).expanduser().resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    cfg["data_root"] = str(data_root)


def main():  # pragma: no cover
    @hydra.main(config_path="../conf", config_name="base_para_breezy", version_base=None)
    def _run(cfg):
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
        _prepare_para_paths(cfg)
        logging.info("data_root: %s", cfg.get("data_root"))
        ParaPipeline(cfg).run()

    print("Finish Config Matching")
    _run()


if __name__ == "__main__":
    main()

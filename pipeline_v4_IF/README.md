---
language:
- en
license: mit
task_categories:
- audio-text-to-text
tags:
- speech
- instruction-following
- spoken-language-model
- TASTE
configs:
- config_name: default
  data_files:
  - split: train
    path: data/shuffled_train_part_*.parquet
  - split: dev
    path: data/shuffled_dev.parquet
- config_name: training
  data_files:
  - split: train
    path: data_training/shuffled_train_part_*.parquet
  - split: dev
    path: data_training/shuffled_dev.parquet
dataset_info:
- config_name: default
  features:
  - name: idx
    dtype: string
  - name: system_prompt
    dtype: string
  - name: meta
    struct:
    - name: instruction_type
      dtype: string
  - name: instruction_audio
    dtype: audio
  - name: response_audio
    dtype: audio
  - name: message
    list:
    - name: role
      dtype: string
    - name: text
      dtype: string
    - name: timestamp_range
      sequence: int64
    - name: para_tags
      dtype: string
- config_name: training
  features:
  - name: idx
    dtype: string
  - name: system_prompt
    dtype: string
  - name: meta
    struct:
    - name: instruction_type
      dtype: string
  - name: message
    list:
    - name: role
      dtype: string
    - name: text
      dtype: string
    - name: audio
      dtype: binary
    - name: timestamp_range
      sequence: int64
    - name: para_tags
      dtype: string
---

# TASTE-IF-SFT-48K

A 48K-example English **speech-oriented Instruction-Following (IF)** SFT dataset,
built for fine-tuning Spoken Language Models (SLMs) such as
[TASTE](https://github.com/) on basic content-level instruction following
(as a precursor to paralinguistic / voice-style instruction following, e.g. VStyle-style tasks).

## Motivation

VStyle-like prompts (e.g. *"With a very slow pace, please count from one to five."*)
combine two abilities:

1. **Content-level instruction following** — understanding *what* to say
   (e.g. "count from one to five" -> "One, two, three, four, five.")
2. **Paralinguistic / acoustic control** — controlling *how* it is said
   (e.g. speaking slowly)

This dataset isolates ability (1): every instruction is a short, speech-friendly
English command **without** any paralinguistic/acoustic conditioning words
(slow, fast, loud, whisper, angry, sad, happy, tone, pitch, volume, etc.).
Each example also includes synthesized speech audio for both the instruction
(user turn) and the expected spoken response (assistant turn), so the dataset
can be used directly for SFT of speech-in/speech-out models.

## Configs

This repo provides two configs of the same 48K examples:

- **`default`** (under `data/`): includes top-level `instruction_audio` / `response_audio`
  columns stored as the standard HF `Audio()` struct (`{"bytes", "path"}`), so the
  Dataset Viewer can play the audio directly in the browser. Recommended for browsing
  and for general-purpose use with `datasets.load_dataset`.
- **`training`** (under `data_training/`): the exact format expected by the TASTE SFT
  data loader (`sft_processor.py`) — each `message` entry carries its own `audio`
  (raw `bytes`, not wrapped in an `Audio()` feature) and there are no top-level
  `instruction_audio`/`response_audio` columns. Use this config if you want to feed
  the data directly into TASTE SFT training:
  ```python
  ds = load_dataset("Jaylin0418/TASTE-IF-SFT-48K", "training")
  ```

## Dataset Structure

- **Splits**: 11 train shards (`shuffled_train_part_0000.parquet` ... `shuffled_train_part_0010.parquet`,
  4,000 examples each, 44,000 total) + 1 dev shard (`shuffled_dev.parquet`, 4,000 examples).
  Total = **48,000 examples**.
- **Format**: Parquet, one row per dialogue example.

### Columns

- `idx` (str): unique example id, prefixed by the instruction-following ability category
  (e.g. `read_aloud_047388`).
- `system_prompt` (str): a generic system prompt describing the IF task (same for every example).
  Provided as a convenience starting point; replace it as needed for your own training setup.
- `meta` (dict):
  - `instruction_type` (str): the instruction-following ability category (see below)
- `instruction_audio` / `response_audio` (`Audio`): the synthesized speech audio for the
  instruction (user turn) and the expected spoken response (assistant turn).
- `message` (list[dict]): a 2-turn conversation, one `user` turn and one `assistant` turn, each with:
  - `role` (str): `"user"` or `"assistant"`
  - `text` (str): the instruction (user) or expected target response (assistant)
  - `timestamp_range` (list[int]): start/end timestep range for this turn within the full dialogue audio
  - `para_tags` (str or null, JSON-encoded): paralinguistic tags applied during TTS synthesis of this
    turn's audio (e.g. `{"emotion": "surprised"}`). **Always `null` for the user turn**, since the
    instruction audio is always synthesized in a neutral/calm voice; only the assistant (response)
    audio is rendered with the requested style.

### Example

```python
{
  "idx": "counting_001234",
  "system_prompt": "You are a helpful spoken assistant. Listen to the user's spoken instruction and respond with only the requested spoken content, following the instruction exactly.",
  "meta": {"instruction_type": "counting"},
  "instruction_audio": {"bytes": b"...", "path": None},
  "response_audio": {"bytes": b"...", "path": None},
  "message": [
    {"role": "user", "text": "Please count from one to five.", "timestamp_range": [0, 3000], "para_tags": "{}"},
    {"role": "assistant", "text": "One, two, three, four, five.", "timestamp_range": [3000, 5500], "para_tags": "{}"}
  ]
}
```

## Ability Categories (`meta.instruction_type`)

```text
read_aloud, counting, sequence, reverse_sequence, listing, repetition, spelling,
number_reading, time_date_reading, format_constraint, negative_constraint,
required_word, word_extraction, replacement, filtering, selection, ordering,
comparison, completion, transformation, short_description, short_generation,
simple_arithmetic, conditional, multi_step
```

## Loading

```python
from datasets import load_dataset

ds = load_dataset("Jaylin0418/TASTE-IF-SFT-48K")
print(ds)
print(ds["train"][0]["message"])
```

## Intended Use

This dataset was used to SFT [TASTE](https://github.com/) on basic spoken
instruction-following ability before evaluating paralinguistic / voice-style
instruction following (VStyle-like tasks). It can be used to fine-tune any
speech-in/speech-out SLM that consumes paired (instruction audio, instruction text,
response audio, response text) examples.

#!/usr/bin/env python3
"""
[ENGLISH VERSION] — see generate_if_data_zh.py for the Traditional Chinese version.

Generate Speech-Oriented Instruction-Following Dataset with Style Dimension.

Each example combines:
  - A content task  (counting, listing, read_aloud, ...)
  - A style modifier (slow / fast / angry / sad / happy / surprised /
                       fearful / disgusted / whisper / none)

Schema:
  {
    "instruction": "In a slow pace, please count from one to five.",
    "target_text": "One, two, three, four, five.",
    "style": "slow",
    "ability": "acoustic_attributes/speed/slow",
    "lang": "en"
  }

IMPORTANT: target_text is ALWAYS plain content only — no style markers.

Usage:
  # Pilot (1,500 examples):
  python generate_if_data.py --mode pilot --api_key sk-...

  # Full (15,000 examples):
  python generate_if_data.py --mode full --api_key sk-...

  # Single category:
  python generate_if_data.py --mode pilot --category counting --api_key sk-...
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# ── Style definitions ─────────────────────────────────────────────────────────

STYLES = ["none", "slow", "fast", "angry", "sad", "happy",
          "surprised", "fearful", "disgusted", "whisper"]

STYLE_ABILITY = {
    "none":      None,   # use content ability
    "slow":      "acoustic_attributes/speed/slow",
    "fast":      "acoustic_attributes/speed/fast",
    "angry":     "acoustic_attributes/emotion/angry",
    "sad":       "acoustic_attributes/emotion/sad",
    "happy":     "acoustic_attributes/emotion/happy",
    "surprised": "acoustic_attributes/emotion/surprised",
    "fearful":   "acoustic_attributes/emotion/fearful",
    "disgusted": "acoustic_attributes/emotion/disgusted",
    "whisper":   "acoustic_attributes/volume/whisper",
}

# ── Allowed ability values ────────────────────────────────────────────────────

ALLOWED_ABILITIES = {
    # content-only (style=none)
    "instruction_following/read_aloud",
    "instruction_following/counting",
    "instruction_following/sequence",
    "instruction_following/reverse_sequence",
    "instruction_following/listing",
    "instruction_following/exact_count",
    "instruction_following/repetition",
    "instruction_following/spelling",
    "instruction_following/number_reading",
    "instruction_following/time_date_reading",
    "instruction_following/format_constraint",
    "instruction_following/negative_constraint",
    "instruction_following/required_word",
    "instruction_following/word_extraction",
    "instruction_following/replacement",
    "instruction_following/filtering",
    "instruction_following/selection",
    "instruction_following/ordering",
    "instruction_following/comparison",
    "instruction_following/completion",
    "instruction_following/transformation",
    "instruction_following/short_description",
    "instruction_following/short_generation",
    "instruction_following/simple_arithmetic",
    "instruction_following/conditional",
    "instruction_following/multi_step",
    # styled
    "acoustic_attributes/speed/slow",
    "acoustic_attributes/speed/fast",
    "acoustic_attributes/emotion/angry",
    "acoustic_attributes/emotion/sad",
    "acoustic_attributes/emotion/happy",
    "acoustic_attributes/emotion/surprised",
    "acoustic_attributes/emotion/fearful",
    "acoustic_attributes/emotion/disgusted",
    "acoustic_attributes/volume/whisper",
}

# ── Category targets ──────────────────────────────────────────────────────────

PILOT_TARGETS = {
    "read_aloud":          300,
    "listing":             225,
    "counting":            180,
    "sequence":            150,
    "repetition":          120,
    "spelling":            120,
    "number_reading":       75,
    "format_constraint":   105,
    "negative_constraint":  75,
    "multi_step":           75,
    "short_description":    25,
    "short_generation":     20,
    "reverse_sequence":     30,
    "exact_count":          30,
    "time_date_reading":    30,
    "required_word":        25,
    "word_extraction":      25,
    "replacement":          25,
    "comparison":           15,
    "completion":           15,
    "simple_arithmetic":    50,
    "conditional":          30,
    "filtering":            10,
    "ordering":             10,
    "transformation":        5,
    "selection":             5,
}

_FULL_TOTAL = 24000
_pilot_total = sum(PILOT_TARGETS.values())
FULL_TARGETS = {k: max(1, round(v * _FULL_TOTAL / _pilot_total))
                for k, v in PILOT_TARGETS.items()}
# adjust read_aloud to hit exactly 20,000
FULL_TARGETS["read_aloud"] += _FULL_TOTAL - sum(FULL_TARGETS.values())

# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a dataset generator for a speech-oriented instruction-following benchmark.

Each example must have EXACTLY these fields: instruction, target_text, style, ability, lang.

Rules:
- Output valid JSONL only. One JSON object per line.
- Do NOT output markdown, comments, numbering, or any explanation.
- lang must always be "en".
- instruction: a short spoken-style command. May contain a style wrapper (slow, fast, angry, etc.) or no style at all.
- target_text: ONLY the plain spoken content. NO style markers, NO parenthetical notes like "(said slowly)", "(whispered)", "(angrily)", "whispers:", etc.
- style: exactly one of: none, slow, fast, angry, sad, happy, surprised, fearful, disgusted, whisper.
- ability: see the per-category instructions.
- target_text must be short (under 30 words). Short description/generation may be up to 30 words.
- No code, URLs, markdown tables, math formulas, or long reasoning.
- No unsafe or inappropriate content.
- No duplicate examples.
- Mix easy (50%), medium (35%), and hard (15%) difficulty.

Style distribution (across all examples in the batch):
  none: ~30%
  slow: ~10%
  fast: ~10%
  angry: ~8%
  sad: ~8%
  happy: ~8%
  surprised: ~7%
  fearful: ~7%
  disgusted: ~7%
  whisper: ~5%

When style is "none", ability = the content ability (e.g. instruction_following/counting).
When style is not "none", ability = the acoustic ability (e.g. acoustic_attributes/speed/slow).

Style wrappers (use diverse phrasings, not just one template):
  slow:      "In a very slow pace, ...", "Please say ... slowly.", "Using a slow delivery, ..."
  fast:      "In a very fast pace, ...", "Please say ... quickly.", "At a fast speed, ..."
  angry:     "In an angry voice, ...", "Say ... angrily.", "With anger, ..."
  sad:       "In a sad voice, ...", "Say ... sadly.", "With a sad tone, ..."
  happy:     "In a cheerful voice, ...", "Say ... happily.", "With a happy tone, ..."
  surprised: "In a surprised voice, ...", "Say ... with surprise.", "Sound surprised as you say ..."
  fearful:   "In a fearful voice, ...", "Say ... with fear.", "Sound scared as you say ..."
  disgusted: "In a disgusted tone, ...", "Say ... with disgust.", "With disgust, ..."
  whisper:   "Please whisper: ...", "Say ... in a whisper.", "Whispering, please say ..."
"""


def make_prompt(category: str, n: int) -> str:
    return CATEGORY_PROMPTS[category].replace("{n}", str(n))


CATEGORY_PROMPTS = {
    "read_aloud": """\
Generate {n} JSONL examples combining STYLE + read_aloud content.

Content: The instruction asks to say/read/repeat/recite a sentence or phrase exactly.
target_text must EXACTLY match the sentence/phrase to be spoken (no style markers).

Content ability (used when style=none): instruction_following/read_aloud
Acoustic ability (used when style≠none): acoustic_attributes/<group>/<style>

Topics: nature, school, family, weather, food, travel, books, music, animals,
city life, ocean, space, sports, daily routines, museums, classrooms, parks.

IMPORTANT: Vary the instruction wording as much as possible. Use diverse phrasings such as:
"Please say ...", "Read this aloud: ...", "Repeat after me: ...", "Read the following: ...",
"Could you say ...", "Speak this sentence: ...", "Say the following out loud: ...",
"Please recite: ...", "Read this sentence: ...", "Go ahead and say: ...",
"Can you read this: ...", "Please read aloud: ...", "Say this out loud: ...",
"Repeat this phrase: ...", "Read the sentence below: ..."

For styled versions, vary how the style is expressed too:
"In a slow pace", "Slowly say", "Read this slowly", "Say it slowly",
"With a happy voice", "Say this happily", "In an excited tone",
"Whisper this", "Say it softly as a whisper", "In a sad voice", "Sadly read"

Examples:
{"instruction": "Please say this sentence: The train arrived before sunset.", "target_text": "The train arrived before sunset.", "style": "none", "ability": "instruction_following/read_aloud", "lang": "en"}
{"instruction": "Read this aloud: The birds sing every morning.", "target_text": "The birds sing every morning.", "style": "none", "ability": "instruction_following/read_aloud", "lang": "en"}
{"instruction": "Repeat after me: Fresh bread and warm soup.", "target_text": "Fresh bread and warm soup.", "style": "none", "ability": "instruction_following/read_aloud", "lang": "en"}
{"instruction": "Could you read this sentence out loud? The library closes at eight.", "target_text": "The library closes at eight.", "style": "none", "ability": "instruction_following/read_aloud", "lang": "en"}
{"instruction": "Slowly read this sentence: The mountains are beautiful.", "target_text": "The mountains are beautiful.", "style": "slow", "ability": "acoustic_attributes/speed/slow", "lang": "en"}
{"instruction": "In a slow pace, please say: The train arrived before sunset.", "target_text": "The train arrived before sunset.", "style": "slow", "ability": "acoustic_attributes/speed/slow", "lang": "en"}
{"instruction": "Say this quickly: I love reading books.", "target_text": "I love reading books.", "style": "fast", "ability": "acoustic_attributes/speed/fast", "lang": "en"}
{"instruction": "In an angry voice, read: The museum opens at nine.", "target_text": "The museum opens at nine.", "style": "angry", "ability": "acoustic_attributes/emotion/angry", "lang": "en"}
{"instruction": "Whisper the following: Fresh bread and warm soup.", "target_text": "Fresh bread and warm soup.", "style": "whisper", "ability": "acoustic_attributes/volume/whisper", "lang": "en"}
{"instruction": "With a sad tone, recite this sentence: The last train has gone.", "target_text": "The last train has gone.", "style": "sad", "ability": "acoustic_attributes/emotion/sad", "lang": "en"}
{"instruction": "Happily say: Today is a wonderful day!", "target_text": "Today is a wonderful day!", "style": "happy", "ability": "acoustic_attributes/emotion/happy", "lang": "en"}

Output JSONL only.
""",

    "counting": """\
Generate {n} JSONL examples combining STYLE + counting content.

Content subtypes:
- count from N to M (forward)
- count backward from N to M
- count by twos / threes / fives
- first N even numbers
- first N odd numbers

target_text must use word form (One, two, three...) and be verifiably correct.
target_text must NOT contain style markers.

Content ability (style=none): instruction_following/counting
Acoustic ability (style≠none): acoustic_attributes/<group>/<style>

IMPORTANT: Vary instruction wording. Use diverse phrasings such as:
"Count from ... to ...", "Please count ...", "Say the numbers from ... to ...",
"Count up to ...", "Count down from ...", "Go ahead and count ...",
"Can you count ...", "Read out the numbers ...", "Say each number from ..."

Examples:
{"instruction": "Count from one to five.", "target_text": "One, two, three, four, five.", "style": "none", "ability": "instruction_following/counting", "lang": "en"}
{"instruction": "Please say the numbers from one to five.", "target_text": "One, two, three, four, five.", "style": "none", "ability": "instruction_following/counting", "lang": "en"}
{"instruction": "Go ahead and count down from five to one.", "target_text": "Five, four, three, two, one.", "style": "none", "ability": "instruction_following/counting", "lang": "en"}
{"instruction": "Slowly count from one to five.", "target_text": "One, two, three, four, five.", "style": "slow", "ability": "acoustic_attributes/speed/slow", "lang": "en"}
{"instruction": "In an angry voice, count backward from five to one.", "target_text": "Five, four, three, two, one.", "style": "angry", "ability": "acoustic_attributes/emotion/angry", "lang": "en"}
{"instruction": "Whisper the even numbers from two to ten.", "target_text": "Two, four, six, eight, ten.", "style": "whisper", "ability": "acoustic_attributes/volume/whisper", "lang": "en"}
{"instruction": "Say the first five odd numbers quickly.", "target_text": "One, three, five, seven, nine.", "style": "fast", "ability": "acoustic_attributes/speed/fast", "lang": "en"}

Output JSONL only.
""",

    "sequence": """\
Generate {n} JSONL examples combining STYLE + fixed sequence content.

Content subtypes:
- days of the week (all or partial range, starting from any day)
- weekdays only / weekend only
- months of the year (all or from X to Y)
- alphabet ranges
- seasons
- ordinal numbers
- morning / afternoon / evening / night

Content ability (style=none): instruction_following/sequence
Acoustic ability (style≠none): acoustic_attributes/<group>/<style>

Examples:
{"instruction": "Please recite the seven days of the week.", "target_text": "Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday.", "style": "none", "ability": "instruction_following/sequence", "lang": "en"}
{"instruction": "In a fast pace, recite the days of the week.", "target_text": "Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday.", "style": "fast", "ability": "acoustic_attributes/speed/fast", "lang": "en"}
{"instruction": "Please say the months from January to June in a sad voice.", "target_text": "January, February, March, April, May, June.", "style": "sad", "ability": "acoustic_attributes/emotion/sad", "lang": "en"}

Output JSONL only.
""",

    "reverse_sequence": """\
Generate {n} JSONL examples combining STYLE + reverse sequence content.

Content: say a fixed sequence backward (days, months, alphabet, etc.)

Content ability (style=none): instruction_following/reverse_sequence
Acoustic ability (style≠none): acoustic_attributes/<group>/<style>

Examples:
{"instruction": "Please say the alphabet from G back to A.", "target_text": "G, F, E, D, C, B, A.", "style": "none", "ability": "instruction_following/reverse_sequence", "lang": "en"}
{"instruction": "In a fearful voice, say the months from June back to January.", "target_text": "June, May, April, March, February, January.", "style": "fearful", "ability": "acoustic_attributes/emotion/fearful", "lang": "en"}

Output JSONL only.
""",

    "listing": """\
Generate {n} JSONL examples combining STYLE + listing content.

Content: name or list N items from a category.
target_text must contain exactly the requested number of items (no style markers).

Content ability (style=none): instruction_following/listing
Acoustic ability (style≠none): acoustic_attributes/<group>/<style>

Diverse categories: fruits, vegetables, drinks, kitchen items, classroom objects,
animals, vehicles, clothes, sports, musical instruments, colors, shapes, planets, etc.

Examples:
{"instruction": "Please list three fruits.", "target_text": "Apples, bananas, and oranges.", "style": "none", "ability": "instruction_following/listing", "lang": "en"}
{"instruction": "In a happy voice, name three animals.", "target_text": "Dogs, cats, and rabbits.", "style": "happy", "ability": "acoustic_attributes/emotion/happy", "lang": "en"}
{"instruction": "Please whisper three things found in a kitchen.", "target_text": "A spoon, a plate, and a cup.", "style": "whisper", "ability": "acoustic_attributes/volume/whisper", "lang": "en"}
{"instruction": "Quickly name four colors.", "target_text": "Red, blue, green, and yellow.", "style": "fast", "ability": "acoustic_attributes/speed/fast", "lang": "en"}

Output JSONL only.
""",

    "exact_count": """\
Generate {n} JSONL examples combining STYLE + exact-count listing.

Content: instruction says "exactly N" items; target_text must have exactly that many.

Content ability (style=none): instruction_following/exact_count
Acoustic ability (style≠none): acoustic_attributes/<group>/<style>

Examples:
{"instruction": "Please give exactly two examples of fruit.", "target_text": "Apples and bananas.", "style": "none", "ability": "instruction_following/exact_count", "lang": "en"}
{"instruction": "In a surprised voice, name exactly three animals that can fly.", "target_text": "Eagles, bats, and butterflies.", "style": "surprised", "ability": "acoustic_attributes/emotion/surprised", "lang": "en"}

Output JSONL only.
""",

    "repetition": """\
Generate {n} JSONL examples combining STYLE + repetition content.

Content subtypes:
- repeat a word N times
- repeat a phrase N times
- repeat a sentence exactly
- say the first / last / second word

Content ability (style=none): instruction_following/repetition
Acoustic ability (style≠none): acoustic_attributes/<group>/<style>

Examples:
{"instruction": "Please repeat the word apple five times.", "target_text": "Apple, apple, apple, apple, apple.", "style": "none", "ability": "instruction_following/repetition", "lang": "en"}
{"instruction": "In an angry voice, repeat the word no three times.", "target_text": "No, no, no.", "style": "angry", "ability": "acoustic_attributes/emotion/angry", "lang": "en"}
{"instruction": "Please whisper this phrase three times: good night.", "target_text": "Good night, good night, good night.", "style": "whisper", "ability": "acoustic_attributes/volume/whisper", "lang": "en"}
{"instruction": "In a happy voice, repeat the sentence: Today is a great day.", "target_text": "Today is a great day.", "style": "happy", "ability": "acoustic_attributes/emotion/happy", "lang": "en"}

Output JSONL only.
""",

    "spelling": """\
Generate {n} JSONL examples combining STYLE + spelling content.

Content subtypes:
- spell a word (letter by letter, comma-separated, uppercase)
- spell a word backward
- say a code letter by letter

Content ability (style=none): instruction_following/spelling
Acoustic ability (style≠none): acoustic_attributes/<group>/<style>

Examples:
{"instruction": "Please spell the word garden.", "target_text": "G, A, R, D, E, N.", "style": "none", "ability": "instruction_following/spelling", "lang": "en"}
{"instruction": "In a slow pace, please spell the word planet.", "target_text": "P, L, A, N, E, T.", "style": "slow", "ability": "acoustic_attributes/speed/slow", "lang": "en"}
{"instruction": "Please whisper the word music, spelled out.", "target_text": "M, U, S, I, C.", "style": "whisper", "ability": "acoustic_attributes/volume/whisper", "lang": "en"}

Output JSONL only.
""",

    "number_reading": """\
Generate {n} JSONL examples combining STYLE + number reading content.

Content subtypes:
- digit by digit (407 → "Four, zero, seven.")
- full number word (58 → "Fifty-eight.")
- year reading (2026 → "Twenty twenty-six.")
- price reading
- decimal reading

Content ability (style=none): instruction_following/number_reading
Acoustic ability (style≠none): acoustic_attributes/<group>/<style>

Examples:
{"instruction": "Please say the number 407 digit by digit.", "target_text": "Four, zero, seven.", "style": "none", "ability": "instruction_following/number_reading", "lang": "en"}
{"instruction": "In a surprised voice, read this year aloud: 2026.", "target_text": "Twenty twenty-six.", "style": "surprised", "ability": "acoustic_attributes/emotion/surprised", "lang": "en"}

Output JSONL only.
""",

    "time_date_reading": """\
Generate {n} JSONL examples combining STYLE + time/date reading.

Content: read a time or date aloud.

Content ability (style=none): instruction_following/time_date_reading
Acoustic ability (style≠none): acoustic_attributes/<group>/<style>

Examples:
{"instruction": "Please read this time aloud: 7:30.", "target_text": "Seven thirty.", "style": "none", "ability": "instruction_following/time_date_reading", "lang": "en"}
{"instruction": "In a slow pace, read this date aloud: March 5.", "target_text": "March fifth.", "style": "slow", "ability": "acoustic_attributes/speed/slow", "lang": "en"}

Output JSONL only.
""",

    "format_constraint": """\
Generate {n} JSONL examples combining STYLE + format constraint.

Content: answer with only one word, exactly N words, start with X, end with Y, etc.

Content ability (style=none): instruction_following/format_constraint
Acoustic ability (style≠none): acoustic_attributes/<group>/<style>

Examples:
{"instruction": "Answer with only one word: What color is the sky?", "target_text": "Blue.", "style": "none", "ability": "instruction_following/format_constraint", "lang": "en"}
{"instruction": "In a happy voice, answer with only one word: What is your favorite fruit?", "target_text": "Mango.", "style": "happy", "ability": "acoustic_attributes/emotion/happy", "lang": "en"}
{"instruction": "Please answer in exactly three words: What do you do when tired?", "target_text": "Take a rest.", "style": "none", "ability": "instruction_following/format_constraint", "lang": "en"}

Output JSONL only.
""",

    "negative_constraint": """\
Generate {n} JSONL examples combining STYLE + negative constraint.

Content: list or say something, but FORBID specific items.
target_text must not contain forbidden items.

Content ability (style=none): instruction_following/negative_constraint
Acoustic ability (style≠none): acoustic_attributes/<group>/<style>

Examples:
{"instruction": "Please name three fruits, but do not mention apples.", "target_text": "Bananas, oranges, and grapes.", "style": "none", "ability": "instruction_following/negative_constraint", "lang": "en"}
{"instruction": "In an angry voice, name three animals but not cat or dog.", "target_text": "Rabbit, horse, and elephant.", "style": "angry", "ability": "acoustic_attributes/emotion/angry", "lang": "en"}

Output JSONL only.
""",

    "required_word": """\
Generate {n} JSONL examples combining STYLE + required word.

Content: say a sentence that must include a specific word.

Content ability (style=none): instruction_following/required_word
Acoustic ability (style≠none): acoustic_attributes/<group>/<style>

Examples:
{"instruction": "Please say a sentence that includes the word ocean.", "target_text": "The ocean looks calm today.", "style": "none", "ability": "instruction_following/required_word", "lang": "en"}
{"instruction": "In a sad voice, say a sentence with the word rain.", "target_text": "The rain falls all day long.", "style": "sad", "ability": "acoustic_attributes/emotion/sad", "lang": "en"}

Output JSONL only.
""",

    "word_extraction": """\
Generate {n} JSONL examples combining STYLE + word extraction.

Content: extract first, last, or Nth word from a given sentence.

Content ability (style=none): instruction_following/word_extraction
Acoustic ability (style≠none): acoustic_attributes/<group>/<style>

Examples:
{"instruction": "Please say only the first word of this sentence: The flowers bloom in spring.", "target_text": "The.", "style": "none", "ability": "instruction_following/word_extraction", "lang": "en"}
{"instruction": "In a slow pace, say only the last word: The train stopped near the station.", "target_text": "Station.", "style": "slow", "ability": "acoustic_attributes/speed/slow", "lang": "en"}

Output JSONL only.
""",

    "replacement": """\
Generate {n} JSONL examples combining STYLE + word replacement.

Content: replace one word with another in a given sentence.

Content ability (style=none): instruction_following/replacement
Acoustic ability (style≠none): acoustic_attributes/<group>/<style>

Examples:
{"instruction": "Please replace cat with dog: The cat is sleeping.", "target_text": "The dog is sleeping.", "style": "none", "ability": "instruction_following/replacement", "lang": "en"}
{"instruction": "In a disgusted tone, replace morning with evening: I walk in the morning.", "target_text": "I walk in the evening.", "style": "disgusted", "ability": "acoustic_attributes/emotion/disgusted", "lang": "en"}

Output JSONL only.
""",

    "filtering": """\
Generate {n} JSONL examples combining STYLE + filtering.

Content: from a mixed list, say only items of the requested category.

Content ability (style=none): instruction_following/filtering
Acoustic ability (style≠none): acoustic_attributes/<group>/<style>

Examples:
{"instruction": "From apple, carrot, and banana, please say only the fruits.", "target_text": "Apple and banana.", "style": "none", "ability": "instruction_following/filtering", "lang": "en"}
{"instruction": "In a surprised voice, from Monday, June, and Friday, say only the days of the week.", "target_text": "Monday and Friday.", "style": "surprised", "ability": "acoustic_attributes/emotion/surprised", "lang": "en"}

Output JSONL only.
""",

    "selection": """\
Generate {n} JSONL examples combining STYLE + selection.

Content: choose the correct item from given options.

Content ability (style=none): instruction_following/selection
Acoustic ability (style≠none): acoustic_attributes/<group>/<style>

Examples:
{"instruction": "Choose the fruit: chair, apple, or shoe.", "target_text": "Apple.", "style": "none", "ability": "instruction_following/selection", "lang": "en"}
{"instruction": "In a fearful voice, which is an animal: river, tiger, or cloud?", "target_text": "Tiger.", "style": "fearful", "ability": "acoustic_attributes/emotion/fearful", "lang": "en"}

Output JSONL only.
""",

    "ordering": """\
Generate {n} JSONL examples combining STYLE + ordering.

Content: put given items in a requested order.

Content ability (style=none): instruction_following/ordering
Acoustic ability (style≠none): acoustic_attributes/<group>/<style>

Examples:
{"instruction": "Please put these numbers in order from smallest to largest: seven, two, five.", "target_text": "Two, five, seven.", "style": "none", "ability": "instruction_following/ordering", "lang": "en"}
{"instruction": "In a fast pace, put these months in calendar order: May, January, March.", "target_text": "January, March, May.", "style": "fast", "ability": "acoustic_attributes/speed/fast", "lang": "en"}

Output JSONL only.
""",

    "comparison": """\
Generate {n} JSONL examples combining STYLE + comparison.

Content: compare two things and say which is bigger/smaller/longer/shorter/earlier.

Content ability (style=none): instruction_following/comparison
Acoustic ability (style≠none): acoustic_attributes/<group>/<style>

Examples:
{"instruction": "Please say which is bigger: seven or three.", "target_text": "Seven is bigger.", "style": "none", "ability": "instruction_following/comparison", "lang": "en"}
{"instruction": "In a happy voice, say which word is longer: apple or watermelon.", "target_text": "Watermelon is longer.", "style": "happy", "ability": "acoustic_attributes/emotion/happy", "lang": "en"}

Output JSONL only.
""",

    "completion": """\
Generate {n} JSONL examples combining STYLE + completion.

Content: complete an incomplete sequence or phrase.

Content ability (style=none): instruction_following/completion
Acoustic ability (style≠none): acoustic_attributes/<group>/<style>

Examples:
{"instruction": "Please complete this sequence: Monday, Tuesday, Wednesday.", "target_text": "Thursday.", "style": "none", "ability": "instruction_following/completion", "lang": "en"}
{"instruction": "In a sad voice, complete this phrase: Peanut butter and.", "target_text": "Jelly.", "style": "sad", "ability": "acoustic_attributes/emotion/sad", "lang": "en"}

Output JSONL only.
""",

    "transformation": """\
Generate {n} JSONL examples combining STYLE + grammatical transformation.

Content: change tense, negate, turn into question, pluralize, etc.

Content ability (style=none): instruction_following/transformation
Acoustic ability (style≠none): acoustic_attributes/<group>/<style>

Examples:
{"instruction": "Please change this to past tense: I open the window.", "target_text": "I opened the window.", "style": "none", "ability": "instruction_following/transformation", "lang": "en"}
{"instruction": "In a disgusted tone, make this sentence negative: She likes carrots.", "target_text": "She does not like carrots.", "style": "disgusted", "ability": "acoustic_attributes/emotion/disgusted", "lang": "en"}

Output JSONL only.
""",

    "short_description": """\
Generate {n} JSONL examples combining STYLE + short description.

Content: describe something in one short sentence (8–20 words).

Content ability (style=none): instruction_following/short_description
Acoustic ability (style≠none): acoustic_attributes/<group>/<style>

Examples:
{"instruction": "Please describe a rainy afternoon in one sentence.", "target_text": "Raindrops cover the windows while the sky turns gray.", "style": "none", "ability": "instruction_following/short_description", "lang": "en"}
{"instruction": "In a sad voice, describe a quiet park in one sentence.", "target_text": "The empty swings move slowly in the cold wind.", "style": "sad", "ability": "acoustic_attributes/emotion/sad", "lang": "en"}

Output JSONL only.
""",

    "short_generation": """\
Generate {n} JSONL examples combining STYLE + short generation.

Content: make a short sentence about a topic, or tell a short joke (under 20 words).

Content ability (style=none): instruction_following/short_generation
Acoustic ability (style≠none): acoustic_attributes/<group>/<style>

Examples:
{"instruction": "Please make a short sentence about space.", "target_text": "The stars shine above the moon.", "style": "none", "ability": "instruction_following/short_generation", "lang": "en"}
{"instruction": "In a happy voice, make a short sentence about summer.", "target_text": "The sun warms the beach all day.", "style": "happy", "ability": "acoustic_attributes/emotion/happy", "lang": "en"}
{"instruction": "Please tell a short joke.", "target_text": "Why did the cookie go to the doctor? Because it felt crummy.", "style": "none", "ability": "instruction_following/short_generation", "lang": "en"}

Output JSONL only.
""",

    "simple_arithmetic": """\
Generate {n} JSONL examples combining STYLE + simple arithmetic.

Content: addition, subtraction, multiplication with small numbers (1–20).
target_text must be the correct answer in word form.

Content ability (style=none): instruction_following/simple_arithmetic
Acoustic ability (style≠none): acoustic_attributes/<group>/<style>

Examples:
{"instruction": "Please answer this: What is two plus three?", "target_text": "Five.", "style": "none", "ability": "instruction_following/simple_arithmetic", "lang": "en"}
{"instruction": "In a surprised voice, what is ten minus four?", "target_text": "Six.", "style": "surprised", "ability": "acoustic_attributes/emotion/surprised", "lang": "en"}

Output JSONL only.
""",

    "conditional": """\
Generate {n} JSONL examples combining STYLE + conditional instruction.

Content: a rule + a condition; target_text applies the rule correctly.

Content ability (style=none): instruction_following/conditional
Acoustic ability (style≠none): acoustic_attributes/<group>/<style>

Examples:
{"instruction": "If the word is apple, say fruit. The word is apple.", "target_text": "Fruit.", "style": "none", "ability": "instruction_following/conditional", "lang": "en"}
{"instruction": "In a fearful voice, if the number is even, say even. The number is six.", "target_text": "Even.", "style": "fearful", "ability": "acoustic_attributes/emotion/fearful", "lang": "en"}

Output JSONL only.
""",

    "multi_step": """\
Generate {n} JSONL examples combining STYLE + multi-step instruction.

Content: two-step spoken command; target_text follows both steps in order.

Content ability (style=none): instruction_following/multi_step
Acoustic ability (style≠none): acoustic_attributes/<group>/<style>

Examples:
{"instruction": "First say hello, then count to three.", "target_text": "Hello. One, two, three.", "style": "none", "ability": "instruction_following/multi_step", "lang": "en"}
{"instruction": "In a happy voice, say hello then name two colors.", "target_text": "Hello. Blue and red.", "style": "happy", "ability": "acoustic_attributes/emotion/happy", "lang": "en"}
{"instruction": "Please whisper one fruit, then spell it.", "target_text": "Apple. A, P, P, L, E.", "style": "whisper", "ability": "acoustic_attributes/volume/whisper", "lang": "en"}

Output JSONL only.
""",
}

# ── Validation ────────────────────────────────────────────────────────────────

# Patterns that should NOT appear in target_text
TARGET_STYLE_MARKERS = re.compile(
    r'(\(said\s|\(spoken\s|\(whisper|\bwhispers:\s|'
    r'\(in\s+an?\s+\w+\s+voice\)|\(slowly\)|\(quickly\)|\(angrily\))',
    re.IGNORECASE
)

BAD_CONTENT_RE = re.compile(
    r'(https?://|www\.|```|<[a-z]+>|\$\{|\bmarkdown\b|<\|im_start\|>)',
    re.IGNORECASE
)


def validate_example(ex: dict) -> tuple[bool, str]:
    required = {"instruction", "target_text", "style", "ability", "lang"}
    missing = required - ex.keys()
    if missing:
        return False, f"missing fields: {missing}"

    if not ex["instruction"].strip():
        return False, "empty instruction"
    if not ex["target_text"].strip():
        return False, "empty target_text"
    if ex["lang"] != "en":
        return False, f"lang={ex['lang']}"
    if ex["style"] not in STYLES:
        return False, f"unknown style: {ex['style']}"
    if ex["ability"] not in ALLOWED_ABILITIES:
        return False, f"unknown ability: {ex['ability']}"

    # Cross-check style vs ability
    expected_ability = STYLE_ABILITY[ex["style"]]
    if expected_ability is not None and ex["ability"] != expected_ability:
        return False, f"style={ex['style']} but ability={ex['ability']}"

    instr_words = len(ex["instruction"].split())
    target_words = len(ex["target_text"].split())
    if instr_words > 80:
        return False, "instruction too long"
    if target_words > 35:
        return False, "target_text too long"

    if TARGET_STYLE_MARKERS.search(ex["target_text"]):
        return False, "target_text contains style marker"

    combined = ex["instruction"] + " " + ex["target_text"]
    if BAD_CONTENT_RE.search(combined):
        return False, "contains code/URL/markdown"

    return True, "ok"


def parse_jsonl_response(text: str) -> list[dict]:
    results = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("```"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        if "lang" not in obj:
            obj["lang"] = "en"
        results.append(obj)
    return results


# ── GPT generation ────────────────────────────────────────────────────────────

def call_gpt(client, category: str, n: int) -> list[dict]:
    prompt = make_prompt(category, n)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.9,
        max_tokens=min(16000, n * 130),
    )
    return parse_jsonl_response(response.choices[0].message.content)


def generate_category(
    client,
    category: str,
    target: int,
    raw_dir: Path,
    validated_dir: Path,
    batch_size: int = 150,
) -> list[dict]:

    validated_path = validated_dir / f"{category}.jsonl"
    existing: list[dict] = []
    seen: set[str] = set()

    if validated_path.exists():
        with open(validated_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        ex = json.loads(line)
                        existing.append(ex)
                        seen.add(ex["instruction"].strip().lower())
                    except Exception:
                        pass
        print(f"  [resume] {category}: {len(existing)}/{target} already done")

    if len(existing) >= target:
        print(f"  [skip] {category}: already have {len(existing)} ≥ {target}")
        return existing

    collected = list(existing)
    batch_num = 0

    while len(collected) < target:
        ask_n = min(batch_size, (target - len(collected)) + 30)
        batch_num += 1
        print(f"  [{category}] batch {batch_num}: asking {ask_n}, have {len(collected)}/{target}")

        for attempt in range(3):
            try:
                raw = call_gpt(client, category, ask_n)
                break
            except Exception as e:
                print(f"    API error (attempt {attempt+1}/3): {e}")
                time.sleep(5 * (attempt + 1))
        else:
            print(f"  ERROR: {category} batch {batch_num} failed, skipping")
            break

        raw_path = raw_dir / f"{category}_batch{batch_num:03d}.jsonl"
        with open(raw_path, "w", encoding="utf-8") as f:
            for ex in raw:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        added = 0
        for ex in raw:
            if len(collected) >= target:
                break
            key = ex.get("instruction", "").strip().lower()
            if key in seen:
                continue
            ok, reason = validate_example(ex)
            if not ok:
                continue
            seen.add(key)
            collected.append(ex)
            added += 1

        print(f"    → accepted {added} (total: {len(collected)}/{target})")

        with open(validated_path, "w", encoding="utf-8") as f:
            for ex in collected:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        if added == 0:
            print(f"  WARNING: no new examples in batch {batch_num}, stopping early")
            break

        time.sleep(1)

    return collected[:target]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--category", default=None)
    parser.add_argument("--batch_size", type=int, default=150)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--total", type=int, default=None,
                        help="Override total target count (scales FULL_TARGETS proportionally)")
    args = parser.parse_args()

    if not args.api_key:
        print("ERROR: provide --api_key or set OPENAI_API_KEY")
        sys.exit(1)

    from openai import OpenAI
    client = OpenAI(api_key=args.api_key)

    if not args.output_dir:
        print("ERROR: --output_dir is required"); sys.exit(1)
    base_dir = Path(args.output_dir)
    raw_dir = base_dir / "raw"
    validated_dir = base_dir / "validated"
    raw_dir.mkdir(parents=True, exist_ok=True)
    validated_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "pilot":
        targets = PILOT_TARGETS
    else:
        if args.total and args.total != _FULL_TOTAL:
            scale = args.total / _FULL_TOTAL
            targets = {k: max(1, round(v * scale)) for k, v in FULL_TARGETS.items()}
            targets["read_aloud"] += args.total - sum(targets.values())
        else:
            targets = FULL_TARGETS

    if args.category:
        if args.category not in targets:
            print(f"ERROR: unknown category '{args.category}'")
            sys.exit(1)
        categories = [args.category]
    else:
        categories = list(targets.keys())

    total_target = sum(targets[c] for c in categories)
    print(f"Mode: {args.mode}  |  Categories: {len(categories)}  |  Target: {total_target} examples")
    print(f"Output: {base_dir}\n")

    all_examples: list[dict] = []
    for category in categories:
        print(f"\n{'='*55}")
        print(f"Category: {category}  →  {targets[category]} examples")
        examples = generate_category(
            client, category, targets[category],
            raw_dir, validated_dir, args.batch_size,
        )
        all_examples.extend(examples)
        print(f"  Done: {len(examples)}")

    dataset_path = base_dir / "dataset.jsonl"
    with open(dataset_path, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\n{'='*55}")
    print(f"DONE. Total: {len(all_examples)} examples → {dataset_path}")

    from collections import Counter
    style_counts = Counter(ex["style"] for ex in all_examples)
    print("\nStyle breakdown:")
    for s in STYLES:
        print(f"  {s:<12} {style_counts.get(s, 0):>5}")


if __name__ == "__main__":
    main()

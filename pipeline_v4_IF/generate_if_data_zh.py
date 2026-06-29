#!/usr/bin/env python3
"""
[TRADITIONAL CHINESE VERSION] — see generate_if_data_en.py for the English version.

Generate Speech-Oriented Instruction-Following Dataset with Style Dimension (Traditional Chinese / zh-TW).

Each example combines:
  - A content task  (counting, listing, read_aloud, ...)
  - A style modifier (slow / fast / angry / sad / happy / surprised /
                       fearful / disgusted / whisper / none)

Schema:
  {
    "instruction": "請用很慢的速度，從一數到五。",
    "target_text": "一、二、三、四、五。",
    "style": "slow",
    "ability": "acoustic_attributes/speed/slow",
    "lang": "zh"
  }

IMPORTANT: target_text is ALWAYS plain content only — no style markers.

Usage:
  # Pilot (1,500 examples):
  python generate_if_data_zh.py --mode pilot --api_key sk-...

  # Full (48,000 examples, matches English dataset.jsonl size):
  python generate_if_data_zh.py --mode full --api_key sk-...

  # Single category:
  python generate_if_data_zh.py --mode pilot --category counting --api_key sk-...
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# ── Style definitions ─────────────────────────────────────────────────────────
# NOTE: style keys and ability strings are kept IDENTICAL to the English version
# (so the two languages share the same `ability` taxonomy for downstream training).

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

# ── Category targets (identical distribution to the English version) ──────────

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

_FULL_TOTAL = 48000  # matches the English dataset.jsonl size (48,000 examples)
_pilot_total = sum(PILOT_TARGETS.values())
FULL_TARGETS = {k: max(1, round(v * _FULL_TOTAL / _pilot_total))
                for k, v in PILOT_TARGETS.items()}
# adjust read_aloud to hit exactly 48,000
FULL_TARGETS["read_aloud"] += _FULL_TOTAL - sum(FULL_TARGETS.values())

# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
你是一個語音導向「指令遵循」(instruction-following) 基準資料集的生成器，目標語言為繁體中文(台灣)。

每筆範例必須剛好包含這些欄位：instruction, target_text, style, ability, lang。

規則：
- 只輸出有效的 JSONL，每行一個 JSON object。
- 不要輸出 markdown、註解、編號或任何說明文字。
- lang 一律為 "zh"。
- instruction 與 target_text 一律使用「繁體中文」(台灣用語、用字)，不要用簡體字。
- instruction：簡短、適合口語表達的指令句，可以包含風格包裝(慢、快、生氣等)，也可以完全沒有風格。
- target_text：只包含「要被唸出來的純內容」，不可包含風格標記或括號註解，例如「(慢慢地)」、「(小聲說)」、「(生氣地)」、「悄聲說：」等。
- style：必須是以下其中一個：none, slow, fast, angry, sad, happy, surprised, fearful, disgusted, whisper。
- ability：請參考各類別的說明。
- target_text 必須簡短(30 字以內)。short_description / short_generation 可放寬到 30 字以內。
- 不要包含程式碼、網址、markdown 表格、數學公式或長篇推理。
- 不要包含不安全或不適當的內容。
- 不要重複相同或高度相似的範例。
- 難度混合：簡單(50%)、中等(35%)、困難(15%)。

Style 分布(整個 batch 內大致比例)：
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

當 style 為 "none" 時，ability = 該類別對應的 content ability(例如 instruction_following/counting)。
當 style 不是 "none" 時，ability = 對應的 acoustic ability(例如 acoustic_attributes/speed/slow)。

Style 包裝語句(請多樣化，不要每次都用同一種句型)：
  slow:      「請用很慢的速度，...」、「慢慢地說...」、「用緩慢的語調，...」、「請放慢速度...」
  fast:      「請用很快的速度，...」、「快速地說...」、「用很快的語調，...」、「請加快速度...」
  angry:     「用生氣的語氣，...」、「請憤怒地說...」、「帶著怒氣，...」、「請用不耐煩、生氣的口吻...」
  sad:       「用悲傷的語氣，...」、「請難過地說...」、「帶著哀傷的語調，...」、「請用低落的口吻...」
  happy:     「用開心的語氣，...」、「請愉快地說...」、「帶著快樂的語調，...」、「請用興奮的口吻...」
  surprised: 「用驚訝的語氣，...」、「請驚訝地說...」、「帶著驚訝的語調，...」、「請用不敢相信的口吻...」
  fearful:   「用害怕的語氣，...」、「請恐懼地說...」、「帶著驚恐的語調，...」、「請用緊張害怕的口吻...」
  disgusted: 「用嫌惡的語氣，...」、「請厭惡地說...」、「帶著反感的語調，...」、「請用不屑的口吻...」
  whisper:   「請用悄悄話的方式說...」、「請輕聲說...」、「用很小的聲音說...」、「悄悄地告訴我...」
"""


def make_prompt(category: str, n: int) -> str:
    return CATEGORY_PROMPTS[category].replace("{n}", str(n))


CATEGORY_PROMPTS = {
    "read_aloud": """\
請生成 {n} 筆 JSONL 範例，結合「風格(style)」與「朗讀(read_aloud)」內容。

內容：指令要求說出/唸出/重複/背誦一句話或一段文字，target_text 必須與被要求唸出的句子「完全一致」(不含風格標記)。

Content ability (style=none)：instruction_following/read_aloud
Acoustic ability (style≠none)：acoustic_attributes/<group>/<style>

主題：自然、學校、家庭、天氣、食物、旅行、書籍、音樂、動物、城市生活、海洋、太空、運動、日常作息、博物館、公園。

重要：請盡量讓 instruction 的句型多樣化，例如：
「請說這句話：...」、「請唸出以下這句話：...」、「跟著我說：...」、「請朗讀以下內容：...」、
「可以說一下這句話嗎：...」、「請把這句話唸出來：...」、「大聲唸出來：...」、
「請複誦這句話：...」、「唸看看這句話：...」、「請把下面這句話說出來：...」

風格的表達方式也要多樣化：
「用很慢的速度」、「慢慢說」、「請放慢速度唸這句話」、
「用開心的語氣」、「請愉快地說這句話」、「用興奮的口吻」、
「請輕聲說」、「用很小的聲音唸出來」、「用悲傷的語氣」、「請難過地唸」

範例：
{"instruction": "請說這句話：火車在日落前就到站了。", "target_text": "火車在日落前就到站了。", "style": "none", "ability": "instruction_following/read_aloud", "lang": "zh"}
{"instruction": "請唸出以下這句話：每天早上都能聽到鳥叫聲。", "target_text": "每天早上都能聽到鳥叫聲。", "style": "none", "ability": "instruction_following/read_aloud", "lang": "zh"}
{"instruction": "跟著我說：剛出爐的麵包配上熱湯。", "target_text": "剛出爐的麵包配上熱湯。", "style": "none", "ability": "instruction_following/read_aloud", "lang": "zh"}
{"instruction": "可以把這句話大聲唸出來嗎：圖書館晚上八點關門。", "target_text": "圖書館晚上八點關門。", "style": "none", "ability": "instruction_following/read_aloud", "lang": "zh"}
{"instruction": "請用很慢的速度唸這句話：山上的風景真美。", "target_text": "山上的風景真美。", "style": "slow", "ability": "acoustic_attributes/speed/slow", "lang": "zh"}
{"instruction": "用很慢的速度，請說：火車在日落前就到站了。", "target_text": "火車在日落前就到站了。", "style": "slow", "ability": "acoustic_attributes/speed/slow", "lang": "zh"}
{"instruction": "請快速地說：我很喜歡看書。", "target_text": "我很喜歡看書。", "style": "fast", "ability": "acoustic_attributes/speed/fast", "lang": "zh"}
{"instruction": "用生氣的語氣唸出：博物館九點開門。", "target_text": "博物館九點開門。", "style": "angry", "ability": "acoustic_attributes/emotion/angry", "lang": "zh"}
{"instruction": "請輕聲說出以下這句話：剛出爐的麵包配上熱湯。", "target_text": "剛出爐的麵包配上熱湯。", "style": "whisper", "ability": "acoustic_attributes/volume/whisper", "lang": "zh"}
{"instruction": "用悲傷的語氣唸這句話：最後一班火車已經開走了。", "target_text": "最後一班火車已經開走了。", "style": "sad", "ability": "acoustic_attributes/emotion/sad", "lang": "zh"}
{"instruction": "請開心地說：今天真是美好的一天！", "target_text": "今天真是美好的一天！", "style": "happy", "ability": "acoustic_attributes/emotion/happy", "lang": "zh"}

只輸出 JSONL。
""",

    "counting": """\
請生成 {n} 筆 JSONL 範例，結合「風格(style)」與「數數(counting)」內容。

內容子類型：
- 從 N 數到 M(順數)
- 從 N 倒數到 M
- 從 N 開始每次加 2 / 3 / 5 連續報數
- 前 N 個偶數
- 前 N 個奇數

target_text 必須使用中文數字寫法(一、二、三...)，並且要正確無誤。
target_text 不可包含風格標記。

Content ability (style=none)：instruction_following/counting
Acoustic ability (style≠none)：acoustic_attributes/<group>/<style>

重要：請讓 instruction 的句型多樣化，例如：
「從...數到...」、「請數一下從...到...」、「請說出從...到...的數字」、
「請數到...」、「請從...開始倒數」、「來，數一下...」、「可以數一下...嗎」

範例：
{"instruction": "請從一數到五。", "target_text": "一、二、三、四、五。", "style": "none", "ability": "instruction_following/counting", "lang": "zh"}
{"instruction": "請說出從一到五的數字。", "target_text": "一、二、三、四、五。", "style": "none", "ability": "instruction_following/counting", "lang": "zh"}
{"instruction": "請從五倒數到一。", "target_text": "五、四、三、二、一。", "style": "none", "ability": "instruction_following/counting", "lang": "zh"}
{"instruction": "請用很慢的速度，從一數到五。", "target_text": "一、二、三、四、五。", "style": "slow", "ability": "acoustic_attributes/speed/slow", "lang": "zh"}
{"instruction": "用生氣的語氣，從五倒數到一。", "target_text": "五、四、三、二、一。", "style": "angry", "ability": "acoustic_attributes/emotion/angry", "lang": "zh"}
{"instruction": "請輕聲說出二到十的偶數。", "target_text": "二、四、六、八、十。", "style": "whisper", "ability": "acoustic_attributes/volume/whisper", "lang": "zh"}
{"instruction": "請快速說出前五個奇數。", "target_text": "一、三、五、七、九。", "style": "fast", "ability": "acoustic_attributes/speed/fast", "lang": "zh"}

只輸出 JSONL。
""",

    "sequence": """\
請生成 {n} 筆 JSONL 範例，結合「風格(style)」與「固定序列(sequence)」內容。

內容子類型：
- 星期一到星期日(全部或部分區間，可從任一天開始)
- 只說平日 / 只說週末
- 一月到十二月(全部或從 X 月到 Y 月)
- 春、夏、秋、冬四季
- 序數(第一、第二、第三...)
- 早上、下午、晚上、深夜(一天的時段)
- 十二生肖(鼠、牛、虎、兔、龍、蛇、馬、羊、猴、雞、狗、豬)

Content ability (style=none)：instruction_following/sequence
Acoustic ability (style≠none)：acoustic_attributes/<group>/<style>

範例：
{"instruction": "請背誦一週的七天。", "target_text": "星期一、星期二、星期三、星期四、星期五、星期六、星期日。", "style": "none", "ability": "instruction_following/sequence", "lang": "zh"}
{"instruction": "請用很快的速度背一週的七天。", "target_text": "星期一、星期二、星期三、星期四、星期五、星期六、星期日。", "style": "fast", "ability": "acoustic_attributes/speed/fast", "lang": "zh"}
{"instruction": "用悲傷的語氣，說出一月到六月。", "target_text": "一月、二月、三月、四月、五月、六月。", "style": "sad", "ability": "acoustic_attributes/emotion/sad", "lang": "zh"}
{"instruction": "請說出十二生肖。", "target_text": "鼠、牛、虎、兔、龍、蛇、馬、羊、猴、雞、狗、豬。", "style": "none", "ability": "instruction_following/sequence", "lang": "zh"}

只輸出 JSONL。
""",

    "reverse_sequence": """\
請生成 {n} 筆 JSONL 範例，結合「風格(style)」與「反向序列(reverse sequence)」內容。

內容：將固定序列倒著說出來(星期、月份、季節、十二生肖、序數等)。

Content ability (style=none)：instruction_following/reverse_sequence
Acoustic ability (style≠none)：acoustic_attributes/<group>/<style>

範例：
{"instruction": "請從星期日開始，倒著說到星期一。", "target_text": "星期日、星期六、星期五、星期四、星期三、星期二、星期一。", "style": "none", "ability": "instruction_following/reverse_sequence", "lang": "zh"}
{"instruction": "用害怕的語氣，把六月到一月倒著說出來。", "target_text": "六月、五月、四月、三月、二月、一月。", "style": "fearful", "ability": "acoustic_attributes/emotion/fearful", "lang": "zh"}

只輸出 JSONL。
""",

    "listing": """\
請生成 {n} 筆 JSONL 範例，結合「風格(style)」與「列舉(listing)」內容。

內容：說出某個類別中的 N 個項目，target_text 必須剛好包含要求的數量(不含風格標記)。

Content ability (style=none)：instruction_following/listing
Acoustic ability (style≠none)：acoustic_attributes/<group>/<style>

多樣化類別：水果、蔬菜、飲料、廚房用品、教室物品、動物、交通工具、衣服、運動、樂器、顏色、形狀、星球等。

範例：
{"instruction": "請列出三種水果。", "target_text": "蘋果、香蕉和橘子。", "style": "none", "ability": "instruction_following/listing", "lang": "zh"}
{"instruction": "用開心的語氣，說出三種動物。", "target_text": "狗、貓和兔子。", "style": "happy", "ability": "acoustic_attributes/emotion/happy", "lang": "zh"}
{"instruction": "請輕聲說出三個廚房裡會有的東西。", "target_text": "湯匙、盤子和杯子。", "style": "whisper", "ability": "acoustic_attributes/volume/whisper", "lang": "zh"}
{"instruction": "請快速說出四種顏色。", "target_text": "紅色、藍色、綠色和黃色。", "style": "fast", "ability": "acoustic_attributes/speed/fast", "lang": "zh"}

只輸出 JSONL。
""",

    "exact_count": """\
請生成 {n} 筆 JSONL 範例，結合「風格(style)」與「精確數量列舉(exact count)」內容。

內容：指令要求「剛好 N 個」項目，target_text 必須剛好有那麼多項。

Content ability (style=none)：instruction_following/exact_count
Acoustic ability (style≠none)：acoustic_attributes/<group>/<style>

範例：
{"instruction": "請說出剛好兩種水果。", "target_text": "蘋果和香蕉。", "style": "none", "ability": "instruction_following/exact_count", "lang": "zh"}
{"instruction": "用驚訝的語氣，說出剛好三種會飛的動物。", "target_text": "老鷹、蝙蝠和蝴蝶。", "style": "surprised", "ability": "acoustic_attributes/emotion/surprised", "lang": "zh"}

只輸出 JSONL。
""",

    "repetition": """\
請生成 {n} 筆 JSONL 範例，結合「風格(style)」與「重複(repetition)」內容。

內容子類型：
- 把某個字重複說 N 次
- 把某個詞語重複說 N 次
- 一字不漏地重複某句話
- 說出第一個字、最後一個字或第二個字

Content ability (style=none)：instruction_following/repetition
Acoustic ability (style≠none)：acoustic_attributes/<group>/<style>

範例：
{"instruction": "請把「蘋果」這個詞重複說五次。", "target_text": "蘋果、蘋果、蘋果、蘋果、蘋果。", "style": "none", "ability": "instruction_following/repetition", "lang": "zh"}
{"instruction": "用生氣的語氣，把「不行」重複說三次。", "target_text": "不行、不行、不行。", "style": "angry", "ability": "acoustic_attributes/emotion/angry", "lang": "zh"}
{"instruction": "請輕聲把「晚安」重複說三次。", "target_text": "晚安、晚安、晚安。", "style": "whisper", "ability": "acoustic_attributes/volume/whisper", "lang": "zh"}
{"instruction": "用開心的語氣，重複這句話：今天是美好的一天。", "target_text": "今天是美好的一天。", "style": "happy", "ability": "acoustic_attributes/emotion/happy", "lang": "zh"}

只輸出 JSONL。
""",

    "spelling": """\
請生成 {n} 筆 JSONL 範例，結合「風格(style)」與「逐字唸讀(spelling)」內容。

由於中文沒有字母拼寫，這個類別請改為「一個字一個字慢慢唸出某個詞語」，
每個字之間用頓號「、」分隔，最後加上句號。

內容子類型：
- 一個字一個字唸出某個詞語(例如「蘋果」→「蘋、果。」)
- 把一個詞語倒過來、一個字一個字唸出
- 一個字一個字唸出一個成語或地名

Content ability (style=none)：instruction_following/spelling
Acoustic ability (style≠none)：acoustic_attributes/<group>/<style>

範例：
{"instruction": "請把「花園」這個詞，一個字一個字唸出來。", "target_text": "花、園。", "style": "none", "ability": "instruction_following/spelling", "lang": "zh"}
{"instruction": "用很慢的速度，一個字一個字唸出「星球」。", "target_text": "星、球。", "style": "slow", "ability": "acoustic_attributes/speed/slow", "lang": "zh"}
{"instruction": "請輕聲一個字一個字唸出「音樂」。", "target_text": "音、樂。", "style": "whisper", "ability": "acoustic_attributes/volume/whisper", "lang": "zh"}
{"instruction": "請把「平安」倒過來，一個字一個字唸出來。", "target_text": "安、平。", "style": "none", "ability": "instruction_following/spelling", "lang": "zh"}

只輸出 JSONL。
""",

    "number_reading": """\
請生成 {n} 筆 JSONL 範例，結合「風格(style)」與「數字唸法(number reading)」內容。

內容子類型：
- 一個數字一個位數一個位數地唸(407 → 「四、零、七。」)
- 用整體中文數字唸法唸出完整數字(58 → 「五十八。」)
- 年份唸法(2026 → 「二零二六年。」或「兩千零二十六年。」)
- 價格唸法(例如 150 元 → 「一百五十元。」)
- 小數唸法(例如 3.5 → 「三點五。」)

Content ability (style=none)：instruction_following/number_reading
Acoustic ability (style≠none)：acoustic_attributes/<group>/<style>

範例：
{"instruction": "請把 407 這個數字，一個位數一個位數地唸出來。", "target_text": "四、零、七。", "style": "none", "ability": "instruction_following/number_reading", "lang": "zh"}
{"instruction": "用驚訝的語氣，唸出這個年份：2026。", "target_text": "二零二六年。", "style": "surprised", "ability": "acoustic_attributes/emotion/surprised", "lang": "zh"}
{"instruction": "請唸出這個價格：150 元。", "target_text": "一百五十元。", "style": "none", "ability": "instruction_following/number_reading", "lang": "zh"}

只輸出 JSONL。
""",

    "time_date_reading": """\
請生成 {n} 筆 JSONL 範例，結合「風格(style)」與「時間/日期唸法(time/date reading)」內容。

內容：唸出一個時間或日期。

Content ability (style=none)：instruction_following/time_date_reading
Acoustic ability (style≠none)：acoustic_attributes/<group>/<style>

範例：
{"instruction": "請唸出這個時間：7:30。", "target_text": "七點半。", "style": "none", "ability": "instruction_following/time_date_reading", "lang": "zh"}
{"instruction": "用很慢的速度，唸出這個日期：3 月 5 日。", "target_text": "三月五號。", "style": "slow", "ability": "acoustic_attributes/speed/slow", "lang": "zh"}

只輸出 JSONL。
""",

    "format_constraint": """\
請生成 {n} 筆 JSONL 範例，結合「風格(style)」與「格式限制(format constraint)」內容。

內容：要求只用一個字回答、剛好用 N 個字回答、用某個字開頭/結尾等。

Content ability (style=none)：instruction_following/format_constraint
Acoustic ability (style≠none)：acoustic_attributes/<group>/<style>

範例：
{"instruction": "請只用一個字回答：天空是什麼顏色？", "target_text": "藍色。", "style": "none", "ability": "instruction_following/format_constraint", "lang": "zh"}
{"instruction": "用開心的語氣，只用一個字回答：你最喜歡的水果是什麼？", "target_text": "芒果。", "style": "happy", "ability": "acoustic_attributes/emotion/happy", "lang": "zh"}
{"instruction": "請用剛好三個字回答：累的時候你會做什麼？", "target_text": "去休息。", "style": "none", "ability": "instruction_following/format_constraint", "lang": "zh"}

只輸出 JSONL。
""",

    "negative_constraint": """\
請生成 {n} 筆 JSONL 範例，結合「風格(style)」與「否定限制(negative constraint)」內容。

內容：列舉或回答時，要求「不能提到」某些特定項目，target_text 不可包含被禁止的項目。

Content ability (style=none)：instruction_following/negative_constraint
Acoustic ability (style≠none)：acoustic_attributes/<group>/<style>

範例：
{"instruction": "請說出三種水果，但不要提到蘋果。", "target_text": "香蕉、橘子和葡萄。", "style": "none", "ability": "instruction_following/negative_constraint", "lang": "zh"}
{"instruction": "用生氣的語氣，說出三種動物，但不要提到貓或狗。", "target_text": "兔子、馬和大象。", "style": "angry", "ability": "acoustic_attributes/emotion/angry", "lang": "zh"}

只輸出 JSONL。
""",

    "required_word": """\
請生成 {n} 筆 JSONL 範例，結合「風格(style)」與「必含詞語(required word)」內容。

內容：說出一句話，必須包含指定的詞語。

Content ability (style=none)：instruction_following/required_word
Acoustic ability (style≠none)：acoustic_attributes/<group>/<style>

範例：
{"instruction": "請說一句包含「海洋」這個詞的句子。", "target_text": "今天的海洋看起來很平靜。", "style": "none", "ability": "instruction_following/required_word", "lang": "zh"}
{"instruction": "用悲傷的語氣，說一句包含「雨」這個字的句子。", "target_text": "雨整天下個不停。", "style": "sad", "ability": "acoustic_attributes/emotion/sad", "lang": "zh"}

只輸出 JSONL。
""",

    "word_extraction": """\
請生成 {n} 筆 JSONL 範例，結合「風格(style)」與「詞語擷取(word extraction)」內容。

內容：從給定的句子中，說出第一個字、最後一個字，或第 N 個字/詞。

Content ability (style=none)：instruction_following/word_extraction
Acoustic ability (style≠none)：acoustic_attributes/<group>/<style>

範例：
{"instruction": "請只說出這句話的第一個字：花朵在春天盛開。", "target_text": "花。", "style": "none", "ability": "instruction_following/word_extraction", "lang": "zh"}
{"instruction": "用很慢的速度，說出這句話的最後一個字：火車在車站附近停了下來。", "target_text": "來。", "style": "slow", "ability": "acoustic_attributes/speed/slow", "lang": "zh"}

只輸出 JSONL。
""",

    "replacement": """\
請生成 {n} 筆 JSONL 範例，結合「風格(style)」與「詞語替換(replacement)」內容。

內容：把給定句子中的某個詞替換成另一個詞。

Content ability (style=none)：instruction_following/replacement
Acoustic ability (style≠none)：acoustic_attributes/<group>/<style>

範例：
{"instruction": "請把「貓」換成「狗」：貓正在睡覺。", "target_text": "狗正在睡覺。", "style": "none", "ability": "instruction_following/replacement", "lang": "zh"}
{"instruction": "用嫌惡的語氣，把「早上」換成「晚上」：我早上去散步。", "target_text": "我晚上去散步。", "style": "disgusted", "ability": "acoustic_attributes/emotion/disgusted", "lang": "zh"}

只輸出 JSONL。
""",

    "filtering": """\
請生成 {n} 筆 JSONL 範例，結合「風格(style)」與「篩選(filtering)」內容。

內容：從一組混合的項目中，只說出屬於指定類別的項目。

Content ability (style=none)：instruction_following/filtering
Acoustic ability (style≠none)：acoustic_attributes/<group>/<style>

範例：
{"instruction": "從蘋果、紅蘿蔔、香蕉中，請只說出水果。", "target_text": "蘋果和香蕉。", "style": "none", "ability": "instruction_following/filtering", "lang": "zh"}
{"instruction": "用驚訝的語氣，從星期一、六月、星期五中，只說出星期幾。", "target_text": "星期一和星期五。", "style": "surprised", "ability": "acoustic_attributes/emotion/surprised", "lang": "zh"}

只輸出 JSONL。
""",

    "selection": """\
請生成 {n} 筆 JSONL 範例，結合「風格(style)」與「選擇(selection)」內容。

內容：從給定的選項中，選出正確的一項。

Content ability (style=none)：instruction_following/selection
Acoustic ability (style≠none)：acoustic_attributes/<group>/<style>

範例：
{"instruction": "請選出水果：椅子、蘋果，還是鞋子？", "target_text": "蘋果。", "style": "none", "ability": "instruction_following/selection", "lang": "zh"}
{"instruction": "用害怕的語氣，請選出動物：河流、老虎，還是雲？", "target_text": "老虎。", "style": "fearful", "ability": "acoustic_attributes/emotion/fearful", "lang": "zh"}

只輸出 JSONL。
""",

    "ordering": """\
請生成 {n} 筆 JSONL 範例，結合「風格(style)」與「排序(ordering)」內容。

內容：把給定的項目按照要求的順序排列。

Content ability (style=none)：instruction_following/ordering
Acoustic ability (style≠none)：acoustic_attributes/<group>/<style>

範例：
{"instruction": "請把這些數字從小到大排列：七、二、五。", "target_text": "二、五、七。", "style": "none", "ability": "instruction_following/ordering", "lang": "zh"}
{"instruction": "用很快的速度，把這些月份按照月曆順序排列：五月、一月、三月。", "target_text": "一月、三月、五月。", "style": "fast", "ability": "acoustic_attributes/speed/fast", "lang": "zh"}

只輸出 JSONL。
""",

    "comparison": """\
請生成 {n} 筆 JSONL 範例，結合「風格(style)」與「比較(comparison)」內容。

內容：比較兩個東西，說出哪個比較大/小/長/短/早。

Content ability (style=none)：instruction_following/comparison
Acoustic ability (style≠none)：acoustic_attributes/<group>/<style>

範例：
{"instruction": "請說出哪個數字比較大：七還是三？", "target_text": "七比較大。", "style": "none", "ability": "instruction_following/comparison", "lang": "zh"}
{"instruction": "用開心的語氣，說出哪個詞比較長：蘋果還是西瓜？", "target_text": "西瓜比較長。", "style": "happy", "ability": "acoustic_attributes/emotion/happy", "lang": "zh"}

只輸出 JSONL。
""",

    "completion": """\
請生成 {n} 筆 JSONL 範例，結合「風格(style)」與「補全(completion)」內容。

內容：補完一個不完整的序列或句子。

Content ability (style=none)：instruction_following/completion
Acoustic ability (style≠none)：acoustic_attributes/<group>/<style>

範例：
{"instruction": "請補完這個序列：星期一、星期二、星期三。", "target_text": "星期四。", "style": "none", "ability": "instruction_following/completion", "lang": "zh"}
{"instruction": "用悲傷的語氣，補完這個句子：屋漏偏逢。", "target_text": "連夜雨。", "style": "sad", "ability": "acoustic_attributes/emotion/sad", "lang": "zh"}

只輸出 JSONL。
""",

    "transformation": """\
請生成 {n} 筆 JSONL 範例，結合「風格(style)」與「語法轉換(transformation)」內容。

內容子類型(中文版改用以下轉換方式)：
- 把肯定句改成否定句
- 把句子改成疑問句
- 在句尾加上「了」表示動作已完成
- 把句子改成「正在...」表示動作正在進行
- 把單數名詞改成「...們」表示複數

Content ability (style=none)：instruction_following/transformation
Acoustic ability (style≠none)：acoustic_attributes/<group>/<style>

範例：
{"instruction": "請把這句話改成否定句：她喜歡吃紅蘿蔔。", "target_text": "她不喜歡吃紅蘿蔔。", "style": "none", "ability": "instruction_following/transformation", "lang": "zh"}
{"instruction": "用嫌惡的語氣，把這句話改成疑問句：他要去公園。", "target_text": "他要去公園嗎？", "style": "disgusted", "ability": "acoustic_attributes/emotion/disgusted", "lang": "zh"}

只輸出 JSONL。
""",

    "short_description": """\
請生成 {n} 筆 JSONL 範例，結合「風格(style)」與「簡短描述(short description)」內容。

內容：用一句話(8 到 20 字)描述某個情境或事物。

Content ability (style=none)：instruction_following/short_description
Acoustic ability (style≠none)：acoustic_attributes/<group>/<style>

範例：
{"instruction": "請用一句話描述下雨的午後。", "target_text": "雨滴打在窗上，天空一片灰暗。", "style": "none", "ability": "instruction_following/short_description", "lang": "zh"}
{"instruction": "用悲傷的語氣，描述一個安靜的公園。", "target_text": "空蕩的鞦韆在冷風中輕輕搖晃。", "style": "sad", "ability": "acoustic_attributes/emotion/sad", "lang": "zh"}

只輸出 JSONL。
""",

    "short_generation": """\
請生成 {n} 筆 JSONL 範例，結合「風格(style)」與「簡短生成(short generation)」內容。

內容：針對某個主題寫一句話，或講一個簡短的笑話(20 字以內)。

Content ability (style=none)：instruction_following/short_generation
Acoustic ability (style≠none)：acoustic_attributes/<group>/<style>

範例：
{"instruction": "請針對「太空」說一句話。", "target_text": "星星在月亮上方閃閃發光。", "style": "none", "ability": "instruction_following/short_generation", "lang": "zh"}
{"instruction": "用開心的語氣，針對「夏天」說一句話。", "target_text": "太陽把整片沙灘曬得暖暖的。", "style": "happy", "ability": "acoustic_attributes/emotion/happy", "lang": "zh"}
{"instruction": "請講一個簡短的笑話。", "target_text": "為什麼餅乾要去看醫生？因為它覺得自己很「酥」弱。", "style": "none", "ability": "instruction_following/short_generation", "lang": "zh"}

只輸出 JSONL。
""",

    "simple_arithmetic": """\
請生成 {n} 筆 JSONL 範例，結合「風格(style)」與「簡單算術(simple arithmetic)」內容。

內容：1 到 20 之間的加法、減法、乘法。target_text 必須是正確答案，並用中文數字寫法。

Content ability (style=none)：instruction_following/simple_arithmetic
Acoustic ability (style≠none)：acoustic_attributes/<group>/<style>

範例：
{"instruction": "請回答：二加三等於多少？", "target_text": "五。", "style": "none", "ability": "instruction_following/simple_arithmetic", "lang": "zh"}
{"instruction": "用驚訝的語氣，十減四等於多少？", "target_text": "六。", "style": "surprised", "ability": "acoustic_attributes/emotion/surprised", "lang": "zh"}

只輸出 JSONL。
""",

    "conditional": """\
請生成 {n} 筆 JSONL 範例，結合「風格(style)」與「條件指令(conditional)」內容。

內容：給一個規則加上一個條件，target_text 必須正確套用該規則。

Content ability (style=none)：instruction_following/conditional
Acoustic ability (style≠none)：acoustic_attributes/<group>/<style>

範例：
{"instruction": "如果這個詞是「蘋果」，請說「水果」。這個詞是「蘋果」。", "target_text": "水果。", "style": "none", "ability": "instruction_following/conditional", "lang": "zh"}
{"instruction": "用害怕的語氣，如果這個數字是偶數，請說「偶數」。這個數字是六。", "target_text": "偶數。", "style": "fearful", "ability": "acoustic_attributes/emotion/fearful", "lang": "zh"}

只輸出 JSONL。
""",

    "multi_step": """\
請生成 {n} 筆 JSONL 範例，結合「風格(style)」與「多步驟指令(multi-step)」內容。

內容：包含兩個步驟的口語指令，target_text 必須依序完成兩個步驟。

Content ability (style=none)：instruction_following/multi_step
Acoustic ability (style≠none)：acoustic_attributes/<group>/<style>

範例：
{"instruction": "先說「你好」，然後從一數到三。", "target_text": "你好。一、二、三。", "style": "none", "ability": "instruction_following/multi_step", "lang": "zh"}
{"instruction": "用開心的語氣，先說「你好」，然後說出兩種顏色。", "target_text": "你好。藍色和紅色。", "style": "happy", "ability": "acoustic_attributes/emotion/happy", "lang": "zh"}
{"instruction": "請輕聲說出一種水果，然後一個字一個字唸出它的名字。", "target_text": "蘋果。蘋、果。", "style": "whisper", "ability": "acoustic_attributes/volume/whisper", "lang": "zh"}

只輸出 JSONL。
""",
}

# ── Validation ────────────────────────────────────────────────────────────────

# Patterns that should NOT appear in target_text (style markers / parenthetical notes)
TARGET_STYLE_MARKERS = re.compile(
    r'(\(慢慢地\)|\(小聲說\)|\(生氣地\)|\(悲傷地\)|\(開心地\)|\(輕聲\)|'
    r'悄聲說[:：]|低聲說[:：]|\(用.{0,6}語氣\)|\(用.{0,6}的口吻\))'
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
    if ex["lang"] != "zh":
        return False, f"lang={ex['lang']}"
    if ex["style"] not in STYLES:
        return False, f"unknown style: {ex['style']}"
    if ex["ability"] not in ALLOWED_ABILITIES:
        return False, f"unknown ability: {ex['ability']}"

    # Cross-check style vs ability
    expected_ability = STYLE_ABILITY[ex["style"]]
    if expected_ability is not None and ex["ability"] != expected_ability:
        return False, f"style={ex['style']} but ability={ex['ability']}"

    # Chinese text length: count characters instead of whitespace-separated words
    instr_chars = len(ex["instruction"])
    target_chars = len(ex["target_text"])
    if instr_chars > 120:
        return False, "instruction too long"
    if target_chars > 60:
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
            obj["lang"] = "zh"
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

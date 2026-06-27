# TASTE2-Data-Pipeline

台灣口音合成對話資料生成 pipeline，用於產生 TASTE2 SFT 訓練資料。

包含四種資料集的完整可重現流程：從 LLM 文字生成 → TTS 合成 → Parquet 轉換（可直接送入 TASTE2 SFT 訓練）。

---

## 產生的四種資料集

| 資料集 | Pipeline | 說明 |
|--------|----------|------|
| **tc** | `pipeline_v1_tc/` | 基礎台灣口音多主題對話，無副語言控制 |
| **tc_emo** | `pipeline_v1_tc/` （`--with-emotion`） | 同 tc，但加入情緒標籤 variant |
| **para** | `pipeline_v2_para/` | 使用者在對話中請求語速/音量/音調/情緒調整 |
| **variant** | `pipeline_v3_variant/` | 同一劇本的多副語言風格 variant（emotion × 6、speed、whisper） |

---

## 環境需求

### 文字生成（全部 pipeline 共用）

```bash
pip install -r requirements.txt
cp .env.example .env
# 填入 OPENROUTER_API_KEY
```

需要 [OpenRouter](https://openrouter.ai/) API key，用於呼叫 LLM（gpt-4o-mini、llama-3.3-70b）。

### TTS 合成（BreezyVoice，tc / para / variant）

```bash
# 建議使用獨立 conda 環境
conda env create -f env/environment.py310.from-history.yml
conda activate breezyvoice_py310

# 或用 bootstrap 腳本（NCHC 環境）
bash env/bootstrap_breezyvoice_py310.sh
```

BreezyVoice model 會在第一次執行時自動從 HuggingFace 下載（`MediaTek-Research/BreezyVoice-300M`）。

需額外 clone BreezyVoice 原始碼 repo：

```bash
git clone https://github.com/MediaTek-Research/BreezyVoice ~/BreezyVoice
# 或設定環境變數指向你的 clone 位置：
export BREEZYVOICE_REPO_DIR=/path/to/BreezyVoice
```

**Reference audio**：

- `tc` pipeline 使用 [Mozilla Common Voice zh-TW](https://commonvoice.mozilla.org/zh-TW/datasets) 作為 TTS ref audio。下載後放至：
  ```
  ~/ref_audio/cv-corpus/zh-TW/validated.tsv
  ~/ref_audio/cv-corpus/zh-TW/clips/
  ```
- `para` / `variant` pipeline 使用 ElevenLabs 情緒音檔（自備，按情緒分類），放至 `~/ref_audio/eleven_lab_emotion/`，結構：
  ```
  eleven_lab_emotion/
  ├── male/
  │   ├── ranbir_1.wav, ranbir_2.wav, ranbir_3.wav
  │   ├── roger_1.wav ...
  │   └── ...
  └── female/
      ├── river_1.wav ...
      └── ...
  ```
- `variant` IndexTTS-2 引擎需額外準備 `~/whisper_emo_ref.wav`（whisper 風格參考音）。

各路徑可透過環境變數覆蓋：`REF_AUDIO_ROOT`、`WHISPER_EMO_REF`、`INDEXTTS_DIR`（詳見 `pipeline_v3_variant/run_variant_tts.py`）。

conf/*.yaml 中的路徑使用 OmegaConf `${oc.env:HOME}` / `${oc.env:USER}` resolver 自動取得目前使用者的 `$HOME` / `$USER`，clone 後不需手動修改路徑。

---

## Pipeline 1：tc / tc_emo

產生基礎台灣口音多主題對話。

```
pipeline_v1_tc/
├── conf/
│   ├── base_v2_breezy.yaml     # 主設定（LLM、TTS、主題列表）
│   └── no_emotion_breezy.yaml  # 無情緒標籤版本
├── generate_txt.sh             # Step 1：LLM 文字生成
├── generate_tts.sh             # Step 2：TTS 合成
├── run_multi_topic_txt_workers.py
├── run_topic_txt.py
├── run_topic_tts.py
├── syn_ver2_breezy.py          # 核心生成邏輯
└── slurm/                      # NCHC SLURM job scripts
```

### 執行

**Step 1：文字生成**

```bash
cd pipeline_v1_tc

# tc（無情緒）
./generate_txt.sh --workers 10 --per-topic-count 220 --output-root-base /work/$USER

# tc_emo（加情緒標籤）
./generate_txt.sh --workers 10 --per-topic-count 220 --with-emotion --output-root-base /work/$USER
```

**Step 2：TTS 合成**

```bash
# tc
RUN_ROOT=/work/$USER/synthetic_dataset_TC ./generate_tts.sh --gpus 0,1,2,3

# tc_emo
RUN_ROOT=/work/$USER/synthetic_dataset_TC_emo ./generate_tts.sh --gpus 0,1,2,3
```

**SLURM（NCHC）**

```bash
sbatch slurm/tts_multi_gpu_all_topics.job
```

---

## Pipeline 2：para

產生使用者在對話中請求副語言控制（語速/音量/音調/情緒）的對話。

```
pipeline_v2_para/
├── conf/
│   └── base_para_breezy.yaml   # 主設定（含 para 控制維度定義）
├── generate_para_txt.sh        # Step 1：LLM 文字生成
├── generate_para_tts.sh        # Step 2：TTS 合成
├── run_topic_para_tts.py
├── syn_para_breezy.py          # 核心生成邏輯
└── slurm/
│   └── para_gen.job
```

### 執行

```bash
cd pipeline_v2_para

# Step 1：文字生成
./generate_para_txt.sh --workers 5 --per-topic-count 100 --output-root-base /work/$USER

# Step 2：TTS 合成
RUN_ROOT=/work/$USER/syn_para_TC ./generate_para_tts.sh --gpus 0,1,2,3 --topics-per-gpu 11
```

---

## Pipeline 3：variant

在已生成的對話劇本上，套用多種副語言風格（emotion × 6、speed_fast、speed_slow、whisper）產生 variant。

```
pipeline_v3_variant/
├── conf/
│   └── base_variant_breezy.yaml
├── generate_variant_txt.sh     # Step 1：LLM 文字生成（含 variant 劇本）
├── run_variant_tts.py          # Step 2：TTS 合成
└── slurm/
    └── run_variant_tts.job
```

### 執行

```bash
cd pipeline_v3_variant

# Step 1：文字生成
./generate_variant_txt.sh --workers 5 --per-topic-count 18 --output-root-base /work/$USER

# Step 2：TTS 合成（需指定 variant data root）
python run_variant_tts.py --run-root /work/$USER/syn_variant --gpus 0,1,2,3
```

---

## Parquet 轉換（訓練用）

把上述音訊資料轉成可直接送入 TASTE2 SFT 訓練的 parquet 格式（schema：`idx`, `meta`, `message`）。

```
to_parquet/
├── to_parquet.py          # tc / tc_emo → parquet
├── to_parquet_variant.py  # variant → parquet
└── hf_to_parquet.py       # para（HF arrow 格式）→ parquet
```

### tc / tc_emo

```bash
python to_parquet/to_parquet.py \
    --data-dir /work/$USER/synthetic_dataset_TC/TEST_syn_data \
    --out-dir  /work/$USER/parquet_sft_combined \
    --prefix   tc

python to_parquet/to_parquet.py \
    --data-dir /work/$USER/synthetic_dataset_TC_emo/TEST_syn_data \
    --out-dir  /work/$USER/parquet_sft_combined \
    --prefix   tc_emo
```

### variant

```bash
python to_parquet/to_parquet_variant.py \
    --data-dir /work/$USER/syn_variant/variant \
    --out-dir  /work/$USER/parquet_sft_combined \
    --prefix   variant
```

### para

```bash
# para 先用 HF datasets 格式存，再轉 parquet
python to_parquet/hf_to_parquet.py \
    --data-dir /work/$USER/syn_para_TC/huggingface_dataset_para/dataset \
    --out-dir  /work/$USER/parquet_sft_combined \
    --prefix   para
```

輸出的 parquet schema：

```json
{
  "idx": "conversation_id",
  "meta": { "para_dimension": "...", "topic": "...", "scenario_id": "...", "scenario_description": "..." },
  "message": [
    { "role": "system",    "text": "...", "audio": null,   "timestamp_range": null },
    { "role": "user",      "text": "...", "audio": <bytes>, "timestamp_range": [start_ms, end_ms] },
    { "role": "assistant", "text": "...", "audio": <bytes>, "timestamp_range": [start_ms, end_ms] }
  ]
}
```

---

## 主題列表（41 個）

Art, Books, Cars, Celebrities, Coding, Cooking, Education, Events, Fashion, Fitness, Finance, Food, Gaming, Gardening, Health, History, Hobbies, Holidays, Home, Languages, Makeup, Movies, Music, Nature, News, Pets, Philosophy, Photography, Podcasts, Politics, Relationships, Science, Shopping, Social Media, Spirituality, Sports, Technology, Traditions, Travel, Weather, Work

---

## Citation

本 pipeline 用於產生 TASTE2-8B 的 SFT 訓練資料。
Model：[Jaylin0418/TASTE2-8B-ZH-SFT](https://huggingface.co/Jaylin0418/TASTE2-8B-ZH-SFT)
Dataset：[Jaylin0418/TASTE2-SFT-Mandarin-Dataset](https://huggingface.co/datasets/Jaylin0418/TASTE2-SFT-Mandarin-Dataset)

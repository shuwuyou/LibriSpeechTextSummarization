# LibriSpeechTextSummarization

A full‑stack project that ingests raw audio, cleans it, transcribes speech to text, performs NLP (Named‑Entity Recognition and multi‑level summarisation) and serves the results via a minimal Flask web UI.

---

## ✨ Key Features

| Stage                        | Tech                                               | Purpose                                                      |
| ---------------------------- | -------------------------------------------------- | ------------------------------------------------------------ |
| **Audio cleaning**           | `librosa`, `soundfile`                             | Resample to 16 kHz, trim silence, normalise loudness         |
| **Speech recognition**       | **OpenAI Whisper** (`openai‑whisper`, `torch`)     | Accurate multilingual ASR                                    |
| **Text cleaning**            | pure Python                                        | Remove fillers, normalise casing & punctuation               |
| **Named‑Entity Recognition** | **spaCy ℹ️** (English `en_core_web_sm` by default) | Extract `PERSON`, `ORG`, `LOC`, etc.                         |
| **Summarisation**            | **T5‑base** via `transformers`                     | Generate long, short & tiny summaries                        |
| **Web UI**                   | **Flask** + vanilla HTML/CSS                       | Upload audio → view cleaned transcript, entities & summaries |
| **Batch CLI tools**          | Python scripts                                     | Automate each stage for large corpora                        |
| **Config‑driven**            | `configs.yaml`                                     | Centralise all parameters                                    |

---

## 🗂️ Directory Layout

```text
project_root/
├─ config/
│  └─ configs.yaml              # all tunable parameters
├─ data/
│  ├─ raw_audio/                # your unprocessed clips
│  ├─ clean_audio/              # output from audio_cleaning.py
│  ├─ transcripts/              # Whisper + cleaned CSV/parquet
│  └─ nlp/                      # NER & summaries (optional)
├─ src/
│  ├─ audio_cleaning.py         # stage 1
│  ├─ whisper_transcribe.py     # stage 2 (batch)
│  ├─ text_cleaning.py          # stage 3
│  ├─ ner_summarise.py          # stage 4 (spaCy + T5) ★ NEW
│  └─ app.py                    # Flask UI
└─ README.md
```

---

## 🔧 Setup

1. **Clone & create venv**

   ```bash
   git clone …
   cd project_root
   python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
   ```
2. **Install core deps**

   ```bash
   pip install -r requirements.txt
   # or, minimal:
   pip install pyyaml librosa soundfile pandas tqdm \
               openai-whisper torch flask spacy transformers
   python -m spacy download en_core_web_sm
   ```
3. **(Windows) Add FFmpeg to PATH** – Whisper uses it to load audio.

---

## ⚙️ Configuration (`config/configs.yaml`)

```yaml
audio_cleaning:
  input_path: "data/raw_audio"
  output_dir: "data/clean_audio"
  sr: 16000
  silence_db: 25
  target_dbfs: -3.0

whisper_transcribe:
  root_dir: "data/clean_audio"
  output_file: "data/transcripts/dev.parquet"
  model: "small"
  device: "cpu"  # change to "cuda" if GPU

text_cleaning:
  input_file: "data/transcripts/dev.parquet"
  output_csv: "data/transcripts/clean.csv"
  text_col: "text"
  lowercase: true
  remove_non_alpha: false
  remove_fillers: true

ner_summarise:
  input_csv: "data/transcripts/clean.csv"
  output_csv: "data/nlp/ner_summary.csv"
  spacy_model: "en_core_web_sm"
  t5_model: "t5-base"
```

---

## 🚀 Quick Start

```bash
# 1. Clean audio
python src/audio_cleaning.py

# 2. Batch ASR
python src/whisper_transcribe.py

# 3. Clean transcripts
python src/text_cleaning.py

# 4. NER & summarisation (optional)
python src/ner_summarise.py

# 5. Launch web UI
python src/app.py  # http://localhost:5000
```

---

## 📝 NLP Stage Details

### spaCy NER (`ner_summarise.py`)

```python
import spacy, pandas as pd
nlp = spacy.load("en_core_web_sm")
...
```

Extracts entities and appends a JSON list to each row (`entities` column).

### T5 Summarisation

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
...
```

Generates three summary lengths via beam‑search.

---

## 📑 Flask UI Endpoints

| Route | Method | Description                               |
| ----- | ------ | ----------------------------------------- |
| `/`   | GET    | Upload form                               |
| `/`   | POST   | Process uploaded file and display results |

The UI calls **in‑memory versions** of the pipeline (audio cleaning → Whisper → text cleaning → NER → summaries) and shows the final cleaned transcript plus extracted entities/summaries.

---

## 🛠️ Troubleshooting

* **FileNotFoundError: ffmpeg** – install FFmpeg and ensure it’s on `PATH`.
* **CUDA out of memory** – switch Whisper/T5 to `device: cpu` or use a smaller model (`tiny`, `base`).
* **spaCy model not found** – run `python -m spacy download en_core_web_sm`.

---

## 📜 License & Acknowledgements

This project is MIT‑licensed. Built with OpenAI Whisper, Hugging Face Transformers, and spaCy – thanks to those communities for the excellent open‑source tooling.

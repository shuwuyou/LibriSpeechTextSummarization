#!/usr/bin/env python3
# whisper.py  –  LibriSpeech → Whisper transcripts
# ---------------------------------------------------------------
# Config file layout (YAML):
# DATA:
#   PATH:      LibriSpeech/dev-clean
# audio_cleaning:       # ← ignored here
# text_cleaning:
#   input_file: data/transcripts/test.parquet
# ---------------------------------------------------------------
import yaml, pathlib, re, random
import whisper, torch, soundfile as sf, pandas as pd
from tqdm.auto import tqdm

# ------------------ 1. load config ------------------------------------------
CFG_PATH = pathlib.Path("config/configs.yaml")     # adjust if needed

with CFG_PATH.open() as f:
    cfg = yaml.safe_load(f)

ROOT_DIR  = pathlib.Path(cfg["DATA"]["PATH"]).expanduser()
OUT_FILE  = pathlib.Path(cfg["text_cleaning"]["input_file"]).expanduser()

WHISPER_MODEL = "small"                     # change as you like
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
FP16          = (DEVICE == "cuda")
SAMPLE_SIZE   = None                       # e.g. 100 to subsample, None = all

print(f"Root dir   : {ROOT_DIR}")
print(f"Output     : {OUT_FILE}")
print(f"Model/dev  : {WHISPER_MODEL} / {DEVICE}")

# ------------------ 2. collect audio -----------------------------------------
flacs = sorted(ROOT_DIR.rglob("*.flac"))
if SAMPLE_SIZE:
    random.seed(42)
    flacs = random.sample(flacs, k=min(SAMPLE_SIZE, len(flacs)))
print(f"Found {len(flacs)} FLAC files to transcribe")

# ------------------ 3. load Whisper ------------------------------------------
asr = whisper.load_model(WHISPER_MODEL, device=DEVICE, in_memory=False)

def parse_ids(p: pathlib.Path):
    spk, chap, utt = re.match(r"(\d+)-(\d+)-(\d+)\.flac", p.name).groups()
    return int(spk), int(chap), int(utt)

def duration(path: str) -> float:
    info = sf.info(path)
    return info.frames / info.samplerate

records = []
for wav in tqdm(flacs, unit="file"):
    txt  = asr.transcribe(str(wav), fp16=FP16)["text"].strip()
    spk, chap, utt = parse_ids(wav)
    records.append({
        "file":    str(wav),
        "speaker": spk,
        "chapter": chap,
        "utt_id":  utt,
        "seconds": duration(str(wav)),
        "text":    txt,
    })

# ------------------ 4. save ---------------------------------------------------
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(records).to_parquet(OUT_FILE, index=False)
print(f"✓ Saved {len(records)} rows → {OUT_FILE}")

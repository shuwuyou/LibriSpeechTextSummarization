#!/usr/bin/env python3
# transcribe_speech.py  –– minimal 3-column output  (file, seconds, text)
# ----------------------------------------------------------------------
import yaml, pathlib, re, sys, torch, pandas as pd, soundfile as sf
from tqdm.auto import tqdm
import whisper

# ---------- config & paths --------------------------------------------
HERE       = pathlib.Path(__file__).resolve()
ROOT       = HERE.parents[1]                    # repo root
CFG_PATH   = ROOT / "config" / "configs.yaml"

with CFG_PATH.open() as f:
    cfg = yaml.safe_load(f)

def R(p):                                       # resolve relative→absolute
    p = pathlib.Path(p)
    return (ROOT / p).resolve() if not p.is_absolute() else p

AUDIO_DIR  = R(cfg["audio_cleaning"]["output_dir"])
OUT_FILE   = R(cfg["text_cleaning"]["input_file"])

print("Input root :", AUDIO_DIR)
print("Output file:", OUT_FILE)

# ---------- collect .wav -------------------------------------------------
wavs = sorted(AUDIO_DIR.rglob("*.[wW][aA][vV]"))
if not wavs:
    sys.exit("⛔  No .wav files found; check audio_cleaning.output_dir")
print(f"Found {len(wavs)} WAV files")

# ---------- Whisper -----------------------------------------------------
asr   = whisper.load_model("small", device="cuda" if torch.cuda.is_available() else "cpu")
dur   = lambda p: sf.info(p).frames / sf.info(p).samplerate

records = []
for wav in tqdm(wavs, unit="file"):
    text = asr.transcribe(str(wav), fp16=torch.cuda.is_available())["text"].strip()
    records.append({"file": str(wav), "seconds": dur(str(wav)), "text": text})

# ---------- save parquet (3 columns only) -------------------------------
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(records, columns=["file", "seconds", "text"]).to_parquet(OUT_FILE, index=False)
print(f"✓ Saved {len(records)} rows → {OUT_FILE}")

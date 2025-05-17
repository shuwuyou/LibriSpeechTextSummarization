#!/usr/bin/env python3
# transcribe_speech.py  -----------------------------------------------
import yaml, pathlib, re, sys, torch, pandas as pd, soundfile as sf
from tqdm.auto import tqdm
import whisper                       # OpenAI Whisper

# ---------- 1. locate repo root & load config --------------------------
HERE       = pathlib.Path(__file__).resolve()
REPO_ROOT  = HERE.parents[1]                     # one level up from src/
CFG_PATH   = REPO_ROOT / "config" / "configs.yaml"

with CFG_PATH.open() as f:
    cfg = yaml.safe_load(f)

def resolve(path_str: str) -> pathlib.Path:
    p = pathlib.Path(path_str)
    return (REPO_ROOT / p).resolve() if not p.is_absolute() else p

ROOT_DIR = resolve(cfg["audio_cleaning"]["output_dir"])
OUT_FILE = resolve(cfg["text_cleaning"]["input_file"])

print("Input root :", ROOT_DIR)
print("Output file:", OUT_FILE)

# ---------- 2. collect *.wav (case-insensitive) ------------------------
wavs = sorted(ROOT_DIR.rglob("*.[wW][aA][vV]"))
if not wavs:
    sys.exit("⛔  No .wav files found – check audio_cleaning.output_dir")

print(f"Found {len(wavs)} WAV files")

# ---------- 3. load Whisper & transcribe ------------------------------
asr     = whisper.load_model("small", device="cuda" if torch.cuda.is_available() else "cpu")
id_re   = re.compile(r"(\d+)-(\d+)-(\d+)\.wav")

def parse_ids(p: pathlib.Path):
    m = id_re.match(p.name)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else (None, None, None)

def duration(path):  # seconds
    i = sf.info(path)
    return i.frames / i.samplerate

records = []
for wav in tqdm(wavs, unit="file"):
    txt = asr.transcribe(str(wav), fp16=torch.cuda.is_available())["text"].strip()
    spk, chap, utt = parse_ids(wav)
    records.append({
        "file": str(wav), "speaker": spk, "chapter": chap,
        "utt_id": utt, "seconds": duration(str(wav)), "text": txt,
    })

# ---------- 4. save ----------------------------------------------------
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(records).to_parquet(OUT_FILE, index=False)
print(f"✓ Saved {len(records)} rows → {OUT_FILE}")

#!/usr/bin/env python3
# ner_extract.py  ────────────────────────────────────────────────────────────
# • Reads the cleaned-transcript CSV produced earlier
# • Runs spaCy NER (default: en_core_web_trf)
# • Writes one-row-per-entity to  ner.output_entity_csv  in configs.yaml
# ---------------------------------------------------------------------------
import yaml, pathlib, sys, importlib.util, pandas as pd, spacy
from tqdm.auto import tqdm

# ── 1. locate repo-root & read YAML ────────────────────────────────────────
HERE       = pathlib.Path(__file__).resolve()
REPO_ROOT  = HERE.parents[1]                 # project root (has /config, /data)
CFG_PATH   = REPO_ROOT / "config" / "configs.yaml"

with CFG_PATH.open() as f:
    cfg = yaml.safe_load(f)

def rel_to_root(p: str) -> pathlib.Path:
    p = pathlib.Path(p)
    return (REPO_ROOT / p).resolve() if not p.is_absolute() else p

INPUT_CSV = rel_to_root(cfg["text_cleaning"]["output_csv"])
OUT_CSV   = rel_to_root(cfg["ner"]["output_entity_csv"])

MODEL_NAME = cfg.get("ner", {}).get("model", "en_core_web_trf")
BATCH_SIZE = cfg.get("ner", {}).get("batch_size", 64)

print("Transcript CSV :", INPUT_CSV)
print("Entity output  :", OUT_CSV)
print("spaCy model    :", MODEL_NAME)

# ── 2. ensure spaCy model installed ────────────────────────────────────────
if importlib.util.find_spec(MODEL_NAME) is None:
    print(f"Downloading spaCy model {MODEL_NAME} …")
    from spacy.cli import download
    download(MODEL_NAME)

nlp = spacy.load(MODEL_NAME)

# ── 3. load transcript CSV ────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV)

# choose text column automatically
TEXT_COL = (
    "cleaned_text"
    if "cleaned_text" in df.columns
    else cfg["text_cleaning"].get("text_col", "text")
)

if TEXT_COL not in df.columns:
    sys.exit(f"⛔  Column '{TEXT_COL}' not found in {INPUT_CSV}")

texts = df[TEXT_COL].astype(str).tolist()

# meta for traceability
meta_cols = [c for c in ["utt_id", "file", "speaker", "chapter"] if c in df.columns]
meta = df[meta_cols].to_dict(orient="records")

# ── 4. run NER in batches ─────────────────────────────────────────────────
records = []
for doc, meta_row in tqdm(
        zip(nlp.pipe(texts, batch_size=BATCH_SIZE), meta),
        total=len(texts),
        desc="spaCy-NER"):
    for ent in doc.ents:
        rec = {
            "sentence": doc.text,
            "entity":   ent.text,
            "label":    ent.label_,
            "start_char": ent.start_char,
            "end_char":   ent.end_char,
        }
        rec.update(meta_row)              # attach utt_id / file / …
        records.append(rec)

# ── 5. save CSV ───────────────────────────────────────────────────────────
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(records).to_csv(OUT_CSV, index=False)
print(f"✓ Saved {len(records)} entities → {OUT_CSV}")

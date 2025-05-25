#!/usr/bin/env python3
# ner_extract.py  ────────────────────────────────────────────────────────────
# • Reads cleaned_transcripts.csv (cols: file, seconds, text, cleaned_text)
# • Runs spaCy NER (default: en_core_web_trf)
# • Aggregates entities per file (pipe-delimited), leaves blank if none
# • Writes out CSV with file, seconds, text, entities, labels
# ---------------------------------------------------------------------------
import yaml, pathlib, sys, importlib.util, pandas as pd, spacy
from tqdm.auto import tqdm

def run_ner_extraction(config_path: pathlib.Path):
    """
    Main function for NER extraction, now callable with a config path.
    """
    with config_path.open() as f:
        cfg = yaml.safe_load(f)

    # Resolve ROOT based on the config_path
    ROOT = config_path.parents[1]
    def R(p: str) -> pathlib.Path:                  # resolve relative→absolute
        p = pathlib.Path(p)
        return (ROOT / p).resolve() if not p.is_absolute() else p

    INPUT_CSV = R(cfg["text_cleaning"]["output_csv"])        # cleaned_transcripts.csv
    OUT_CSV   = R(cfg["ner"]["output_entity_csv"])           # data/ner_results/…

    MODEL_NAME = cfg.get("ner", {}).get("model", "en_core_web_trf")
    BATCH_SIZE = cfg.get("ner", {}).get("batch_size", 64)

    print("Transcript CSV :", INPUT_CSV)
    print("Entity output  :", OUT_CSV)
    print("spaCy model    :", MODEL_NAME)

    # ── 2. ensure spaCy model present ----------------------------------------
    if importlib.util.find_spec(MODEL_NAME) is None:
        try:
            from spacy.cli import download
            print(f"Downloading spaCy model '{MODEL_NAME}'...")
            download(MODEL_NAME)
        except Exception as e:
            raise RuntimeError(f"Failed to download spaCy model '{MODEL_NAME}': {e}")
    nlp = spacy.load(MODEL_NAME)

    # ── 3. load transcript CSV -----------------------------------------------
    if not INPUT_CSV.exists():
        raise ValueError(f"⛔  Input CSV not found: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    # choose the column to run NER on
    TEXT_COL = "cleaned_text" if "cleaned_text" in df.columns else "text"
    if TEXT_COL not in df.columns:
        raise ValueError(f"⛔  Column '{TEXT_COL}' not found in {INPUT_CSV}")

    # keep only the fields we need for the final merge
    df_trans = df[["file", "seconds", TEXT_COL]].rename(columns={TEXT_COL: "text"})

    # ── 4. run spaCy NER and collect per-entity rows --------------------------
    entity_rows = []
    for doc, (file_path, _) in tqdm(
        zip(nlp.pipe(df_trans["text"].tolist(), batch_size=BATCH_SIZE),
            df_trans[["file", "seconds"]].itertuples(index=False)),
        total=len(df_trans),
        desc="spaCy-NER"
    ):
        for ent in doc.ents:
            entity_rows.append({
                "file":    file_path,
                "entity":  ent.text,
                "label":   ent.label_
            })

    # ── 5. aggregate entities per file ---------------------------------------
    if entity_rows:
        df_ent = pd.DataFrame(entity_rows)
        agg = (
            df_ent
            .groupby("file", as_index=False)
            .agg({
                "entity": lambda ents: "|".join(ents),
                "label":  lambda labs:  "|".join(labs)
            })
        )
    else:
        # no entities found in any, create empty structure
        agg = pd.DataFrame(columns=["file", "entity", "label"])

    # ── 6. merge and fill missing --------------------------------------------
    df_out = df_trans.merge(agg, on="file", how="left")
    df_out["entity"] = df_out["entity"].fillna("")
    df_out["label"]  = df_out["label"].fillna("")

    # ── 7. save --------------------------------------------------------------
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"✓ Saved {len(df_out)} rows → {OUT_CSV}")

if __name__ == "__main__":
    # Original main function logic, now calls the new run_ner_extraction
    config_path_local = pathlib.Path(__file__).resolve().parents[1] / "config" / "configs.yaml"
    try:
        run_ner_extraction(config_path_local)
    except (ValueError, RuntimeError) as e:
        print(e)
        sys.exit(1)


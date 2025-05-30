#!/usr/bin/env python3
# summarizationt5.py  –– Generate multi-granularity T5 summaries
# Reads ner.output_entity_csv and writes summarization.final_output
# ----------------------------------------------------------------------
import yaml
import pathlib
import sys
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm.auto import tqdm # Added tqdm for progress bar

def run_summarization(config_path: pathlib.Path):
    """
    Main function for T5 summarization, now callable with a config path.
    """
    with config_path.open() as f:
        cfg = yaml.safe_load(f)

    # Resolve ROOT based on the config_path
    ROOT = config_path.parents[1]
    def R(p: str) -> pathlib.Path:
        pth = pathlib.Path(p)
        return (ROOT / pth).resolve() if not pth.is_absolute() else pth

    # ── 2. input/output paths ───────────────────────────────────────────────
    INPUT_CSV = R(cfg["ner"]["output_entity_csv"])
    OUT_CSV   = R(cfg["summarization"]["final_output"])

    if not INPUT_CSV.exists():
        raise ValueError(f"⛔  Input CSV not found: {INPUT_CSV}")

    # ── 3. load data ─────────────────────────────────────────────────────────
    df = pd.read_csv(INPUT_CSV)
    for col in ("file","seconds","text","entity","label"):
        if col not in df.columns:
            raise ValueError(f"⛔  Column '{col}' missing in {INPUT_CSV}")

    # ── 4. load T5 model & tokenizer ────────────────────────────────────────
    MODEL_NAME = cfg["summarization"].get("model_name", "google/flan-t5-small")
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
    model      = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

    # ── 5. helper for summarization ─────────────────────────────────────────
    def summarize(text: str, prompt: str, max_new_tokens: int) -> str:
        inp = prompt + text
        inputs = tokenizer(
            inp,
            return_tensors="pt",
            truncation=True,
            max_length=tokenizer.model_max_length
        ).to(device)
        outs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            early_stopping=True
        )
        return tokenizer.decode(outs[0], skip_special_tokens=True).strip()

    # ── 6. generate summaries ────────────────────────────────────────────────
    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating Summaries"):
        text = str(row["text"])
        # prompts
        long_prompt  = "Summarize in detail (multiple paragraphs): "
        short_prompt = "Summarize in one paragraph: "
        tiny_prompt  = "Summarize in one sentence: "
        # generate
        tiny  = summarize(text, tiny_prompt,  32)
        short = summarize(text, short_prompt, 128)
        long  = summarize(text, long_prompt,  256)
        records.append({
            "file":    row["file"],
            "seconds": row["seconds"],
            "text":    text,
            "entity":  row["entity"],
            "label":   row["label"],
            "tiny":    tiny,
            "short":   short,
            "long":    long,
        })

    # ── 7. save final DataFrame ──────────────────────────────────────────────
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(OUT_CSV, index=False)
    print(f"✓ Wrote {len(records)} rows → {OUT_CSV}")

if __name__ == "__main__":
    # Original main function logic, now calls the new run_summarization
    config_path_local = pathlib.Path(__file__).resolve().parents[1] / "config" / "configs.yaml"
    try:
        run_summarization(config_path_local)
    except ValueError as e:
        print(e)
        sys.exit(1)


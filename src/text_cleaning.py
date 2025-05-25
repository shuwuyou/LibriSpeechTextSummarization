#!/usr/bin/env python

import re
import yaml
import pandas as pd
from pathlib import Path
from typing import Optional

def clean_text(text: str, lowercase: bool = True, remove_non_alpha: bool = False, remove_fillers: bool = True) -> str:
    if lowercase:
        text = text.lower()
    if remove_fillers:
        fillers = ["uh", "um", "you know", "like", "i mean", "so", "well"]
        for word in fillers:
            text = re.sub(rf"\\b{word}\\b", "", text)
    if remove_non_alpha:
        text = re.sub(r"[^a-zA-Z0-9\\s]", "", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text

def clean_transcripts(input_path: Path, output_csv: Path, text_col: str = "text", lowercase: bool = True,
                      remove_non_alpha: bool = False, remove_fillers: bool = True) -> pd.DataFrame:
    ext = input_path.suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(input_path)
    elif ext == ".json":
        df = pd.read_json(input_path)
    elif ext == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        raise ValueError("Only .csv, .json, or .parquet files are supported.")

    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in input.")

    df["cleaned_text"] = df[text_col].apply(lambda x: clean_text(str(x), lowercase, remove_non_alpha, remove_fillers))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df

def main():
    config_path = Path(__file__).resolve().parents[1] / "config" / "configs.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)["text_cleaning"]

    input_file = Path(config["input_file"])
    output_csv = Path(config["output_csv"])
    text_col = config.get("text_col", "text")
    lowercase = config.get("lowercase", True)
    remove_non_alpha = config.get("remove_non_alpha", False)
    remove_fillers = config.get("remove_fillers", True)

    clean_transcripts(
        input_path=input_file,
        output_csv=output_csv,
        text_col=text_col,
        lowercase=lowercase,
        remove_non_alpha=remove_non_alpha,
        remove_fillers=remove_fillers
    )

    print(f"✅ Cleaned transcripts saved to {output_csv}")

if __name__ == "__main__":
    main()
    
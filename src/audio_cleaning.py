#!/usr/bin/env python
"""
Audio cleaning script that loads parameters from config/configs.yaml
"""
import csv
import yaml
from pathlib import Path
import librosa
import numpy as np
import soundfile as sf

def trim_silence(samples: np.ndarray, thresh_db: int = 25, frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
    non_silent = librosa.effects.split(samples, top_db=thresh_db, frame_length=frame_length, hop_length=hop_length)
    if len(non_silent) == 0:
        return samples
    start, end = non_silent[0][0], non_silent[-1][1]
    return samples[start:end]

def peak_normalise(samples: np.ndarray, target_dbfs: float = -3.0) -> np.ndarray:
    peak = np.max(np.abs(samples))
    if peak == 0:
        return samples
    target_linear = 10 ** (target_dbfs / 20)
    return samples * (target_linear / peak)

def process_one_file(in_path: Path, out_path: Path, sr: int, silence_db: int, target_dbfs: float) -> float:
    samples, _ = librosa.load(in_path, sr=sr, mono=True)
    samples = trim_silence(samples, thresh_db=silence_db)
    samples = peak_normalise(samples, target_dbfs=target_dbfs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, samples, sr, subtype="PCM_16")
    return len(samples) / sr

def main():
    config_path = Path(__file__).resolve().parents[1] / "config" / "configs.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)["audio_cleaning"]

    input_path = Path(config["input_path"])
    output_dir = Path(config["output_dir"])
    sr = config["sr"]
    silence_db = config["silence_db"]
    target_dbfs = config["target_dbfs"]

    audio_exts = {".wav", ".mp3", ".flac", ".aac", ".m4a", ".ogg"}
    files = []
    if input_path.is_file() and input_path.suffix.lower() in audio_exts:
        files = [input_path]
        rel_paths = [Path(input_path.name)]
    elif input_path.is_dir():
        files = sorted(p for p in input_path.rglob("*") if p.suffix.lower() in audio_exts)
        rel_paths = [f.relative_to(input_path) for f in files]
    else:
        raise SystemExit("⚠️  Input must be a valid audio file or directory containing audio files!")

    if not files:
        raise SystemExit("⚠️  No audio files found!")

    metadata_rows = [("file", "duration_sec")]
    for f, rel_path in zip(files, rel_paths):
        out_path = output_dir / rel_path.with_suffix(".wav")
        dur = process_one_file(f, out_path, sr=sr, silence_db=silence_db, target_dbfs=target_dbfs)
        metadata_rows.append((str(out_path), round(dur, 3)))
        print(f"✓ {rel_path}  →  {out_path.name}  ({dur:.1f}s)")

    csv_path = output_dir / "clean_audio_metadata.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(metadata_rows)

    print(f"\n🏁 Done!  Cleaned files saved to {output_dir}")
    print(f"    Metadata CSV: {csv_path}")

if __name__ == "__main__":
    main()

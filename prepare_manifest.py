"""
Validate and normalize a MELD-style manifest.

Usage:
python preprocessing/prepare_manifest.py --input data/raw/manifest.csv \
    --output data/processed/meld_manifest.csv
"""

import argparse
import pandas as pd

REQUIRED = [
    "dialogue_id", "utterance_id", "speaker", "text",
    "audio_path", "video_path", "label"
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.dropna(subset=["text", "label"]).copy()
    df["label"] = df["label"].astype(str).str.lower().str.strip()

    allowed = {
        "anger", "disgust", "sadness", "joy",
        "neutral", "surprise", "fear"
    }
    df = df[df["label"].isin(allowed)]
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} rows to {args.output}")

if __name__ == "__main__":
    main()

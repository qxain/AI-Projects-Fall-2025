"""
data_prep.py
------------
Download, clean, and split the Kaggle Spam Email Dataset.
"""

import os
import re
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Optional: uncomment if you want to automatically download from Kaggle
# (requires your kaggle.json API key in ~/.kaggle)
# from kaggle.api.kaggle_api_extended import KaggleApi



# Configuration

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

for d in [DATA_RAW, DATA_PROCESSED]:
    d.mkdir(parents=True, exist_ok=True)

RAW_CSV = DATA_RAW / "spam_email_dataset.csv"



# Text cleaning helpers


def basic_clean(text: str) -> str:
    """Simple lowercase and punctuation cleanup."""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " URL ", text)
    text = re.sub(r"\d+", " NUM ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["text"].astype(str).apply(basic_clean)
    df = df.dropna(subset=["text"])
    return df[["text", "label"]]


# Kaggle dataset loader

def load_kaggle_csv(path: Path) -> pd.DataFrame:
    """Load and normalize the Kaggle Spam Email Dataset."""
    df = pd.read_csv(path, encoding="latin-1")
    df.columns = [c.lower() for c in df.columns]

    # Detect the label and text columns
    label_col = [c for c in df.columns if "label" in c or "category" in c][0]
    text_col = [c for c in df.columns if "text" in c or "message" in c][0]

    df = df[[label_col, text_col]].rename(columns={label_col: "label", text_col: "text"})
    df["label"] = df["label"].astype(str).str.lower().map({"spam": 1, "ham": 0})
    print(f"Loaded {len(df)} rows from {path}")
    return df

# Split & save

def save_splits(df: pd.DataFrame, out_dir: Path, test_size=0.2, random_state=42):
    out_dir.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=test_size, random_state=random_state, stratify=df["label"]
    )
    pd.DataFrame({"text": X_train, "label": y_train}).to_csv(out_dir / "train.csv", index=False)
    pd.DataFrame({"text": X_test, "label": y_test}).to_csv(out_dir / "test.csv", index=False)
    print(f"    Saved processed data to: {out_dir}")
    print(f"   Train size: {len(X_train)}, Test size: {len(X_test)}")

# Main routine

def main():
    # Optional: Uncomment this block to auto-download from Kaggle API
    # if not RAW_CSV.exists():
    #     print("Downloading dataset from Kaggle ...")
    #     api = KaggleApi()
    #     api.authenticate()
    #     api.dataset_download_files("jackksoncsie/spam-email-dataset", path=str(DATA_RAW), unzip=True)

    if not RAW_CSV.exists():
        raise FileNotFoundError(f"{RAW_CSV} not found. Download it from Kaggle and place it in data/raw/")

    print("Loading Kaggle dataset ...")
    df = load_kaggle_csv(RAW_CSV)
    df = preprocess_df(df)
    save_splits(df, DATA_PROCESSED)

if __name__ == "__main__":
    main()
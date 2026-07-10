"""Downloads and normalizes the public datasets used to train PhishNet's models.

Email corpus: zefang-liu/phishing-email-dataset on Hugging Face - a combined,
publicly-licensed copy of six classic phishing/spam research corpora (Enron,
Ling-Spam, CEAS_08, Nazario, Nigerian Fraud/419, SpamAssassin). Combining them
gives a spread from obvious mass-market scams to real targeted phishing that
closely mimics legitimate correspondence, instead of training on a single
source the way the original PhishNet model did.

URL corpus: pirocheto/phishing-url on Hugging Face - 11,430 URLs, balanced
50/50 phishing vs. legitimate, used as ground truth for the structural URL
risk model in url_features.py. We only ever use the raw `url` string and
`status` label from it; the dataset's own precomputed features are ignored so
training and serving always run through the exact same feature extractor.
"""
import io
import os

import pandas as pd
import requests

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

EMAIL_DATASET_URL = (
    "https://huggingface.co/datasets/zefang-liu/phishing-email-dataset/"
    "resolve/main/Phishing_Email.csv"
)
URL_DATASET_URLS = {
    "train": (
        "https://huggingface.co/datasets/pirocheto/phishing-url/"
        "resolve/main/data/train.parquet"
    ),
    "test": (
        "https://huggingface.co/datasets/pirocheto/phishing-url/"
        "resolve/main/data/test.parquet"
    ),
}


def _download(url, dest_path):
    if os.path.exists(dest_path):
        return dest_path
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"Downloading {url} -> {dest_path}")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    with open(dest_path, "wb") as f:
        f.write(resp.content)
    return dest_path


def load_email_dataset():
    """Returns a DataFrame with columns: text, label (1=phishing, 0=legit)."""
    path = _download(EMAIL_DATASET_URL, os.path.join(DATA_DIR, "Phishing_Email.csv"))
    df = pd.read_csv(path)
    df = df.rename(columns={"Email Text": "text", "Email Type": "label"})
    df = df.dropna(subset=["text", "label"])
    df["label"] = (df["label"] == "Phishing Email").astype(int)
    df["text"] = df["text"].astype(str)
    df = df.drop_duplicates(subset=["text"])
    return df[["text", "label"]].reset_index(drop=True)


def load_url_dataset():
    """Returns a DataFrame with columns: url, label (1=phishing, 0=legit)."""
    frames = []
    for split, url in URL_DATASET_URLS.items():
        path = _download(url, os.path.join(DATA_DIR, f"phishing_url_{split}.parquet"))
        frames.append(pd.read_parquet(path))
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"status": "label"})
    df["label"] = (df["label"] == "phishing").astype(int)
    df = df.drop_duplicates(subset=["url"])
    return df[["url", "label"]].reset_index(drop=True)


if __name__ == "__main__":
    emails = load_email_dataset()
    urls = load_url_dataset()
    print(f"Email dataset: {len(emails)} rows, {emails['label'].mean():.1%} phishing")
    print(f"URL dataset:   {len(urls)} rows, {urls['label'].mean():.1%} phishing")

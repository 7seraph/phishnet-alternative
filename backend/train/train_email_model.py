"""Trains the email text classifier used by /api/analyze.

Upgrades over the original PhishNet model:
  - Trained on a combined, six-source corpus (Enron, Ling-Spam, CEAS_08,
    Nazario, Nigerian Fraud, SpamAssassin) instead of Enron alone, so it sees
    everything from obvious mass-market scams to real spear-phishing that
    closely imitates legitimate email.
  - TF-IDF now includes bigrams and is followed by class-balanced Logistic
    Regression instead of Multinomial Naive Bayes: better probability
    calibration, and linear coefficients we can use to explain *why* a given
    email was flagged (see explain.py).

Usage:
    python train_email_model.py
"""
import json
import os
import sys

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from datasets import load_email_dataset  # noqa: E402
from heuristics import keyword_hits  # noqa: E402

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "model", "phishing_detector.pkl"
)
METRICS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "model", "phishing_detector.metrics.json"
)


def difficulty_bucket(subject, body):
    hits = keyword_hits(subject, body)
    if hits == 0:
        return "hard (no obvious lexical red flags)"
    if hits == 1:
        return "medium"
    return "easy (multiple obvious red flags)"


def main():
    df = load_email_dataset()
    print(f"Loaded {len(df)} unique emails ({df['label'].mean():.1%} phishing)")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                min_df=3,
                max_df=0.9,
                max_features=40000,
                sublinear_tf=True,
            ),
        ),
        (
            "clf",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                C=10,
                random_state=42,
            ),
        ),
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=["legit", "phishing"], output_dict=True)
    print(classification_report(y_test, y_pred, target_names=["legit", "phishing"]))

    # Break the test set down by how many obvious lexical red flags it
    # contains, so we can see whether accuracy holds up on "hard" phishing
    # that doesn't rely on generic spam-y language.
    buckets = {}
    for text, true_label, pred_label in zip(X_test, y_test, y_pred):
        bucket = difficulty_bucket("", text)
        b = buckets.setdefault(bucket, {"correct": 0, "total": 0})
        b["total"] += 1
        b["correct"] += int(true_label == pred_label)

    print("Accuracy by difficulty bucket (lexical-signal proxy):")
    bucket_report = {}
    for name, b in sorted(buckets.items()):
        acc = b["correct"] / b["total"] if b["total"] else 0
        bucket_report[name] = {"accuracy": round(acc, 4), "n": b["total"]}
        print(f"  {name}: {acc:.1%} ({b['total']} examples)")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(
            {
                "dataset_size": len(df),
                "phishing_ratio": round(float(df["label"].mean()), 4),
                "classification_report": report,
                "difficulty_buckets": bucket_report,
            },
            f,
            indent=2,
        )
    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()

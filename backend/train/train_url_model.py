"""Trains the URL risk model used to score links found inside an email.

This is new: the original PhishNet only classified the email body text and
flagged links with a static suspicious-TLD list. That misses brand
impersonation like `wellsfargo--com--verify.wsipv6.com`, which doesn't use
any suspicious TLD but stuffs the brand name into a fake subdomain. Here we
train a small classifier on purely structural/lexical URL features (see
../url_features.py - the same extractor runs at serve time, so there's no
train/serve skew) using pirocheto/phishing-url, a balanced 11k-URL benchmark.

Usage:
    python train_url_model.py
"""
import json
import os
import sys

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from datasets import load_url_dataset  # noqa: E402
from url_features import FEATURE_NAMES, feature_vector  # noqa: E402

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "model", "url_detector.pkl"
)
METRICS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "model", "url_detector.metrics.json"
)


def main():
    df = load_url_dataset()
    print(f"Loaded {len(df)} unique URLs ({df['label'].mean():.1%} phishing)")

    X = [feature_vector(u) for u in df["url"]]
    y = df["label"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=20,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=["legit", "phishing"], output_dict=True)
    print(classification_report(y_test, y_pred, target_names=["legit", "phishing"]))

    importances = sorted(
        zip(FEATURE_NAMES, clf.feature_importances_), key=lambda p: -p[1]
    )
    print("Top features by importance:")
    for name, imp in importances[:10]:
        print(f"  {name}: {imp:.3f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(
            {
                "dataset_size": len(df),
                "phishing_ratio": round(float(df["label"].mean()), 4),
                "classification_report": report,
                "feature_importances": {n: round(float(i), 4) for n, i in importances},
            },
            f,
            indent=2,
        )
    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()

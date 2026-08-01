"""PhishNet Alternative - local backend for the browser extension.

Serves two locally-trained models (see train/) plus a rule-based "real-time
feedback" layer that doesn't depend on any external LLM:

  - phishing_detector.pkl: TF-IDF + SGDClassifier(loss="log_loss") (i.e.
    Logistic Regression trained via gradient descent, which - unlike
    sklearn's plain LogisticRegression - supports partial_fit), trained on a
    combined six-source email corpus (Enron, Ling-Spam, CEAS_08, Nazario,
    Nigerian Fraud, SpamAssassin) spanning obvious mass-market scams to real
    targeted phishing.
  - url_detector.pkl: a structural/lexical URL risk model (url_features.py)
    used to score every link pulled out of the message, so brand
    impersonation like "wellsfargo--com--verify.wsipv6.com" gets flagged
    even when it doesn't hit a static suspicious-TLD list.

Both predictions feed into a plain-English explanation (explain.py) and a
list of tailored phishing-avoidance tips (tips.py). If a Letta token/agent
are configured via environment variables, an additional AI-generated insight
is appended - otherwise that step is skipped silently.

/api/feedback lets the extension's "Correct the model" button submit a
verified/corrected subject, sender, and body for the currently open message
plus the true label; that example is persisted to train/data/corrections.csv
and a background loop (start_background_retrain_loop) continuously pulls any
rows it hasn't seen yet and "backpropagates" them into the live model with a
partial_fit gradient step, so the model updates in real time - whether the
row came from this request, a manual edit to the CSV, or activity that
happened while the server was down.
"""
import csv
import datetime
import json
import os
import threading
import time

import joblib
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS

from explain import build_explanation, top_contributing_terms
from heuristics import analyze_heuristics, domain_of, extract_links
from tips import build_tips
from url_features import feature_vector

load_dotenv()

app = Flask(__name__)
CORS(app)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
EMAIL_MODEL_PATH = os.path.join(MODEL_DIR, "phishing_detector.pkl")
email_model = joblib.load(EMAIL_MODEL_PATH)

URL_MODEL_PATH = os.path.join(MODEL_DIR, "url_detector.pkl")
url_model = joblib.load(URL_MODEL_PATH) if os.path.exists(URL_MODEL_PATH) else None

CORRECTIONS_PATH = os.path.join(os.path.dirname(__file__), "train", "data", "corrections.csv")
CORRECTIONS_STATE_PATH = os.path.join(MODEL_DIR, "corrections_state.json")
CORRECTIONS_POLL_SECONDS = float(os.getenv("CORRECTIONS_POLL_SECONDS", "15"))

# Guards every read-modify-write of the live model + the corrections offset,
# since both the background loop and /api/feedback requests touch them.
_retrain_lock = threading.Lock()


def append_correction(subject, sender, body_text, label_idx):
    """Persists a user correction as a new labeled training row. Doesn't
    touch the live model itself - apply_pending_corrections() (called right
    after, and again on every background loop tick) is what actually learns
    from it."""
    os.makedirs(os.path.dirname(CORRECTIONS_PATH), exist_ok=True)
    is_new = not os.path.exists(CORRECTIONS_PATH)
    with open(CORRECTIONS_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["timestamp", "subject", "sender", "text", "label"])
        writer.writerow([
            datetime.datetime.utcnow().isoformat(),
            subject,
            sender,
            f"{subject} {body_text}".strip(),
            label_idx,
        ])


def _load_processed_row_count():
    if not os.path.exists(CORRECTIONS_STATE_PATH):
        return 0
    with open(CORRECTIONS_STATE_PATH) as f:
        return json.load(f).get("processed_rows", 0)


def _save_processed_row_count(n):
    with open(CORRECTIONS_STATE_PATH, "w") as f:
        json.dump({"processed_rows": n}, f)


def apply_pending_corrections():
    """Pulls whatever rows of corrections.csv haven't been folded into the
    live model yet - tracked as a row count in corrections_state.json - and
    applies them as one partial_fit batch, then saves the model. Idempotent:
    a tick with nothing new is a cheap no-op, so it's safe to call from both
    the request handler (for instant feedback) and the background loop (as
    a continuous catch-up sweep).
    """
    clf = email_model.named_steps.get("clf")
    if clf is None or not hasattr(clf, "partial_fit"):
        return 0
    if not os.path.exists(CORRECTIONS_PATH):
        return 0

    with _retrain_lock:
        with open(CORRECTIONS_PATH, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        processed = _load_processed_row_count()
        new_rows = rows[processed:]
        if not new_rows:
            return 0

        vectorizer = email_model.named_steps["tfidf"]
        texts = [row["text"] for row in new_rows]
        labels = [int(row["label"]) for row in new_rows]

        clf.partial_fit(vectorizer.transform(texts), labels)
        joblib.dump(email_model, EMAIL_MODEL_PATH)
        _save_processed_row_count(len(rows))

        return len(new_rows)


def _background_retrain_loop():
    while True:
        try:
            applied = apply_pending_corrections()
            if applied:
                app.logger.info(
                    "Live-learning loop applied %d pending correction(s) to the model", applied
                )
        except Exception:
            app.logger.exception("Live-learning loop failed to apply pending corrections")
        time.sleep(CORRECTIONS_POLL_SECONDS)


def start_background_retrain_loop():
    """Starts the loop that keeps the live model in sync with corrections.csv.

    `python app.py` runs with debug=True (see bottom of this file), so
    Werkzeug's reloader re-executes this entire module in a throwaway
    "monitor" process - which never serves a request - before launching the
    real worker process, which already has WERKZEUG_RUN_MAIN=true set by the
    time it starts executing this file. Checking for that env var, rather
    than app.debug (which app.run() itself hasn't set yet at the point this
    is called), is what tells the two apart so the loop doesn't start twice.
    """
    if __name__ == "__main__" and os.environ.get("WERKZEUG_RUN_MAIN") is None:
        return
    threading.Thread(target=_background_retrain_loop, daemon=True).start()


def score_links(body_html):
    links = extract_links(body_html)
    if not links or url_model is None:
        return links, []

    vectors = [feature_vector(link["href"]) for link in links]
    scores = url_model.predict_proba(vectors)[:, 1]
    return links, [
        {"href": link["href"], "text": link["text"], "score": round(float(score), 3)}
        for link, score in zip(links, scores)
    ]


def get_llm_insight(subject, body_text, prediction):
    token = os.getenv("LETTA_TOKEN")
    agent_id = os.getenv("LETTA_AGENT_ID")
    if not token or not agent_id:
        return None
    try:
        from letta_client import Letta

        client = Letta(token=token)
        response = client.agents.messages.create(
            agent_id=agent_id,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"The email below was classified as '{prediction}'. In 2-3 sentences, "
                        f"explain why:\n\nSubject: {subject}\n\n{body_text}"
                    ),
                }
            ],
        )
        return response.messages[-1].content.strip()
    except Exception as exc:  # LLM insight is best-effort; never break the response
        app.logger.warning("Letta insight unavailable: %s", exc)
        return None


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True) or {}
    subject = data.get("subject", "")
    body_text = data.get("bodyText", "")
    body_html = data.get("bodyHtml", "")
    sender = data.get("sender", "")

    if not body_text.strip():
        return jsonify({"error": "No email body provided"}), 400

    combined_text = f"{subject} {body_text}"
    prediction_idx = int(email_model.predict([combined_text])[0])
    prob = email_model.predict_proba([combined_text])[0][prediction_idx]
    prediction = "fake" if prediction_idx == 1 else "real"

    terms = top_contributing_terms(email_model, combined_text, prediction_idx)
    explanation = build_explanation(prediction, float(prob), terms)

    links, link_risks = score_links(body_html)
    reasons = analyze_heuristics(sender, subject, body_text, body_html, link_risks)

    insight = get_llm_insight(subject, body_text, prediction)
    if insight:
        reasons.append({"id": "ai_insight", "message": f"AI insight: {insight}"})

    tips = build_tips([r["id"] for r in reasons], prediction)

    return jsonify(
        {
            "prediction": prediction,
            "confidence": round(float(prob), 3),
            "explanation": explanation,
            "topTerms": terms,
            "reasons": [r["message"] for r in reasons],
            "tips": tips,
            "links": link_risks,
        }
    )


@app.route("/api/feedback", methods=["POST"])
def feedback():
    """Lets the user correct a wrong (or confirm a right) verdict for the
    currently open message's subject/sender/body. The correction is persisted
    to corrections.csv and then immediately folded into the live model
    (apply_pending_corrections runs the same sweep the background loop does,
    so the user sees the effect of their own correction right away instead
    of waiting for the next poll tick).
    """
    data = request.get_json(force=True) or {}
    subject = data.get("subject", "")
    sender = data.get("sender", "")
    body_text = data.get("bodyText", "")
    label = data.get("label")

    if label not in ("fake", "real"):
        return jsonify({"error": "label must be 'fake' or 'real'"}), 400
    if not body_text.strip():
        return jsonify({"error": "No email body provided"}), 400

    label_idx = 1 if label == "fake" else 0
    append_correction(subject, sender, body_text, label_idx)
    applied = apply_pending_corrections()

    if not applied:
        return jsonify({
            "status": "saved",
            "message": (
                "Correction saved for the next retrain, but the live model "
                "doesn't support incremental updates. Run "
                "train/train_email_model.py to apply it now."
            ),
        })

    combined_text = f"{subject} {body_text}"
    updated_prob = float(email_model.predict_proba([combined_text])[0][label_idx])

    return jsonify({
        "status": "updated",
        "message": "Thanks - the model was updated with your correction.",
        "updatedConfidence": round(updated_prob, 3),
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "urlModel": url_model is not None})


start_background_retrain_loop()

if __name__ == "__main__":
    app.run(port=5000, debug=True)

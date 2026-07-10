"""PhishNet Alternative - local backend for the browser extension.

Serves two locally-trained models (see train/) plus a rule-based "real-time
feedback" layer that doesn't depend on any external LLM:

  - phishing_detector.pkl: TF-IDF + Logistic Regression, trained on a
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
"""
import os

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
email_model = joblib.load(os.path.join(MODEL_DIR, "phishing_detector.pkl"))

URL_MODEL_PATH = os.path.join(MODEL_DIR, "url_detector.pkl")
url_model = joblib.load(URL_MODEL_PATH) if os.path.exists(URL_MODEL_PATH) else None


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


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "urlModel": url_model is not None})


if __name__ == "__main__":
    app.run(port=5000, debug=True)

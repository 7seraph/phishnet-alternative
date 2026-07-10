"""PhishNet Alternative - local backend for the browser extension.

Serves the same TF-IDF + Naive Bayes phishing classifier as the original
PhishNet, plus a rule-based "real-time feedback" layer that doesn't depend
on any external LLM. If a Letta token/agent are configured via environment
variables, an additional AI-generated insight is appended - otherwise that
step is skipped silently.
"""
import os
from html.parser import HTMLParser
from urllib.parse import urlparse

import joblib
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS

load_dotenv()

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "phishing_detector.pkl")
model = joblib.load(MODEL_PATH)

URGENCY_WORDS = [
    "urgent", "immediately", "act now", "verify your account", "account suspended",
    "suspend your account", "confirm your identity", "unusual activity",
    "limited time", "act within", "your account will be closed", "final notice",
    "password will expire", "click here",
]

GENERIC_GREETINGS = [
    "dear customer", "dear user", "dear valued customer", "dear account holder",
    "dear member", "dear sir/madam",
]

SENSITIVE_WORDS = [
    "password", "social security", "ssn", "credit card", "bank account",
    "routing number", "one-time code", "otp", "verify your account",
    "pin number",
]

SUSPICIOUS_TLDS = [".ru", ".tk", ".top", ".xyz", ".click", ".gq", ".cf", ".zip", ".rest"]


class _LinkExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self._href = None
        self._text = ""

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self._href = dict(attrs).get("href")
            self._text = ""

    def handle_data(self, data):
        if self._href is not None:
            self._text += data

    def handle_endtag(self, tag):
        if tag == "a" and self._href is not None:
            self.links.append({"href": self._href, "text": self._text.strip()})
            self._href = None
            self._text = ""


def extract_links(html):
    if not html:
        return []
    parser = _LinkExtractor()
    try:
        parser.feed(html)
    except Exception:
        pass
    return [l for l in parser.links if l["href"]]


def domain_of(value):
    if not value:
        return ""
    if "://" not in value:
        value = f"//{value}"
    try:
        return urlparse(value).netloc.lower().split("@")[-1]
    except Exception:
        return ""


def analyze_heuristics(sender, subject, body_text, body_html):
    reasons = []
    text = f"{subject} {body_text}".lower()

    urgency_hits = sorted({w for w in URGENCY_WORDS if w in text})
    if urgency_hits:
        reasons.append("Urgent/pressure language detected: " + ", ".join(urgency_hits[:3]))

    if any(g in text for g in GENERIC_GREETINGS):
        reasons.append("Generic greeting instead of your name - common in mass phishing emails.")

    sensitive_hits = sorted({w for w in SENSITIVE_WORDS if w in text})
    if sensitive_hits:
        reasons.append("Requests sensitive info: " + ", ".join(sensitive_hits[:3]))

    links = extract_links(body_html)

    mismatched = []
    for link in links:
        href_domain = domain_of(link["href"])
        text_domain = domain_of(link["text"]) if "." in link["text"] else ""
        if text_domain and href_domain and text_domain != href_domain:
            mismatched.append(link)
    if mismatched:
        reasons.append(f"{len(mismatched)} link(s) display one destination but point somewhere else.")

    suspicious_links = [l for l in links if domain_of(l["href"]).endswith(tuple(SUSPICIOUS_TLDS))]
    if suspicious_links:
        domains = sorted({domain_of(l["href"]) for l in suspicious_links})
        reasons.append("Link(s) point to uncommon/high-risk domains: " + ", ".join(domains[:3]))

    sender_domain = domain_of(sender.split("@")[-1]) if "@" in sender else ""
    link_domains = {domain_of(l["href"]) for l in links}
    link_domains.discard("")
    if sender_domain and link_domains and sender_domain not in link_domains:
        reasons.append("Sender's domain doesn't match any linked domains in the message.")

    return reasons


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
    prediction_idx = model.predict([combined_text])[0]
    prob = model.predict_proba([combined_text])[0][prediction_idx]
    prediction = "fake" if prediction_idx == 1 else "real"

    reasons = analyze_heuristics(sender, subject, body_text, body_html)
    insight = get_llm_insight(subject, body_text, prediction)
    if insight:
        reasons.append(f"AI insight: {insight}")

    return jsonify(
        {
            "prediction": prediction,
            "confidence": round(float(prob), 3),
            "reasons": reasons,
        }
    )


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(port=5000, debug=True)

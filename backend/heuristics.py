"""Keyword lists and rule-based red-flag detection shared by the backend and
the training scripts (the latter uses these to bucket test emails into easy
vs. hard cases for reporting - see train/train_email_model.py).

Each reason returned by analyze_heuristics() carries an `id` so the API layer
can attach a matching tip from TIPS in app.py without re-parsing message text.
"""
from html.parser import HTMLParser
from urllib.parse import urlparse

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


def keyword_hits(subject, body_text):
    """Lightweight lexical signal count, used to bucket eval examples by
    difficulty during training (0 hits = hard/subtle, 2+ hits = easy/obvious)."""
    text = f"{subject} {body_text}".lower()
    hits = 0
    hits += any(w in text for w in URGENCY_WORDS)
    hits += any(g in text for g in GENERIC_GREETINGS)
    hits += any(w in text for w in SENSITIVE_WORDS)
    return hits


def analyze_heuristics(sender, subject, body_text, body_html, link_risks=None):
    """Returns a list of {id, message} red flags found via simple rules.

    link_risks is an optional list of {href, text, score} for links already
    scored by the URL risk model (see url_features.py / app.py) - when
    provided, flags high-risk links instead of relying on a static TLD list.
    """
    reasons = []
    text = f"{subject} {body_text}".lower()

    urgency_hits = sorted({w for w in URGENCY_WORDS if w in text})
    if urgency_hits:
        reasons.append({
            "id": "urgency",
            "message": "Urgent/pressure language detected: " + ", ".join(urgency_hits[:3]),
        })

    if any(g in text for g in GENERIC_GREETINGS):
        reasons.append({
            "id": "generic_greeting",
            "message": "Generic greeting instead of your name - common in mass phishing emails.",
        })

    sensitive_hits = sorted({w for w in SENSITIVE_WORDS if w in text})
    if sensitive_hits:
        reasons.append({
            "id": "sensitive_info",
            "message": "Requests sensitive info: " + ", ".join(sensitive_hits[:3]),
        })

    links = extract_links(body_html)

    mismatched = []
    for link in links:
        href_domain = domain_of(link["href"])
        text_domain = domain_of(link["text"]) if "." in link["text"] else ""
        if text_domain and href_domain and text_domain != href_domain:
            mismatched.append(link)
    if mismatched:
        reasons.append({
            "id": "link_mismatch",
            "message": f"{len(mismatched)} link(s) display one destination but point somewhere else.",
        })

    if link_risks:
        risky = [l for l in link_risks if l["score"] >= 0.4]
        if risky:
            domains = sorted({domain_of(l["href"]) for l in risky})
            reasons.append({
                "id": "risky_link",
                "message": (
                    f"{len(risky)} link(s) have URL patterns typical of phishing "
                    "(e.g. brand names stuffed into the domain, IP-address links, "
                    "link shorteners): " + ", ".join(d for d in domains[:3] if d)
                ),
            })

    sender_domain = domain_of(sender.split("@")[-1]) if "@" in sender else ""
    link_domains = {domain_of(l["href"]) for l in links}
    link_domains.discard("")
    if sender_domain and link_domains and sender_domain not in link_domains:
        reasons.append({
            "id": "sender_link_mismatch",
            "message": "Sender's domain doesn't match any linked domains in the message.",
        })

    return reasons

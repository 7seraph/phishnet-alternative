"""Structural/lexical feature extraction for a single URL string.

Every feature here is computable from the URL text alone - no DNS lookups,
WHOIS queries, or fetching the destination page. That's a deliberate
constraint: the backend scores links found in an email in real time and
should never visit them, and the exact same function has to run both at
training time (over the pirocheto/phishing-url dataset) and at serve time
(over links pulled out of a Gmail message), so any feature that needs a
network call would introduce train/serve skew anyway.
"""
import re
from urllib.parse import urlparse

SUSPICIOUS_TLDS = {
    "ru", "tk", "top", "xyz", "click", "gq", "cf", "zip", "rest", "work",
    "support", "gdn", "mom", "loan", "men", "date", "stream", "download",
    "racing", "win", "bid", "review", "kim", "country",
}

SHORTENING_SERVICES = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly",
    "rebrand.ly", "cutt.ly", "shorte.st", "rb.gy", "tiny.cc", "s.id",
    "lnkd.in", "trib.al",
}

BRAND_KEYWORDS = [
    "paypal", "apple", "microsoft", "amazon", "google", "netflix", "meta",
    "facebook", "instagram", "bankofamerica", "wellsfargo", "chase",
    "citibank", "irs", "usps", "fedex", "dhl", "ups", "ebay", "linkedin",
    "outlook", "office365", "docusign", "coinbase", "binance", "venmo",
    "zelle", "adobe", "dropbox", "steam", "spotify",
]

FEATURE_NAMES = [
    "length_url", "length_hostname", "has_ip", "nb_dots", "nb_hyphens",
    "nb_at", "nb_qm", "nb_and", "nb_eq", "nb_underscore", "nb_percent",
    "nb_slash", "nb_www", "nb_com", "https_token_in_path",
    "ratio_digits_url", "ratio_digits_host", "punycode", "has_port",
    "nb_subdomains", "prefix_suffix", "shortening_service",
    "suspicious_tld", "brand_in_subdomain", "brand_not_in_domain",
    "avg_word_length_path", "longest_word_length_path", "uses_https",
]

_IP_RE = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")
_WORD_RE = re.compile(r"[a-zA-Z0-9]+")


def _registrable_domain(hostname):
    """Best-effort second-level-domain guess, e.g. 'mail.wellsfargo.com' -> 'wellsfargo'."""
    parts = hostname.split(".")
    if len(parts) < 2:
        return hostname
    return parts[-2]


def extract_url_features(url):
    url = (url or "").strip()
    if "://" not in url:
        url = f"http://{url}"

    try:
        parsed = urlparse(url)
    except Exception:
        parsed = None

    hostname = (parsed.hostname or "") if parsed else ""
    path = (parsed.path or "") if parsed else ""
    scheme = (parsed.scheme or "") if parsed else ""

    subdomain_parts = hostname.split(".")[:-2] if hostname.count(".") >= 2 else []
    subdomain = ".".join(subdomain_parts)
    domain = _registrable_domain(hostname)
    host_and_domain = hostname.replace(subdomain, "", 1) if subdomain else hostname

    words_in_path = _WORD_RE.findall(path)
    tld = hostname.rsplit(".", 1)[-1].lower() if "." in hostname else ""
    registrable = f"{domain}.{tld}".lower()

    digits_url = sum(c.isdigit() for c in url)
    digits_host = sum(c.isdigit() for c in hostname)

    brand_hit_subdomain = any(b in subdomain.lower() for b in BRAND_KEYWORDS)
    brand_hit_anywhere = any(b in url.lower() for b in BRAND_KEYWORDS)
    brand_hit_domain = any(b in domain.lower() for b in BRAND_KEYWORDS)

    features = {
        "length_url": len(url),
        "length_hostname": len(hostname),
        "has_ip": 1 if _IP_RE.match(hostname) else 0,
        "nb_dots": url.count("."),
        "nb_hyphens": url.count("-"),
        "nb_at": url.count("@"),
        "nb_qm": url.count("?"),
        "nb_and": url.count("&"),
        "nb_eq": url.count("="),
        "nb_underscore": url.count("_"),
        "nb_percent": url.count("%"),
        "nb_slash": url.count("/"),
        "nb_www": url.lower().count("www"),
        "nb_com": url.lower().count("com"),
        "https_token_in_path": 1 if "https" in (path + subdomain).lower() else 0,
        "ratio_digits_url": digits_url / len(url) if url else 0.0,
        "ratio_digits_host": digits_host / len(hostname) if hostname else 0.0,
        "punycode": 1 if "xn--" in hostname.lower() else 0,
        "has_port": 1 if parsed and parsed.port else 0,
        "nb_subdomains": len(subdomain_parts),
        "prefix_suffix": 1 if "-" in domain else 0,
        "shortening_service": 1 if registrable in SHORTENING_SERVICES else 0,
        "suspicious_tld": 1 if tld in SUSPICIOUS_TLDS else 0,
        "brand_in_subdomain": 1 if brand_hit_subdomain and not brand_hit_domain else 0,
        "brand_not_in_domain": 1 if brand_hit_anywhere and not brand_hit_domain else 0,
        "avg_word_length_path": (
            sum(len(w) for w in words_in_path) / len(words_in_path) if words_in_path else 0.0
        ),
        "longest_word_length_path": max((len(w) for w in words_in_path), default=0),
        "uses_https": 1 if scheme == "https" else 0,
    }
    return features


def feature_vector(url):
    features = extract_url_features(url)
    return [features[name] for name in FEATURE_NAMES]

# PhishNet - Alternative Version (Browser Extension)

This is a spin-off of [PhishNet](https://github.com/7seraph/phishnet), reworked as a browser
extension: click the toolbar icon while reading an email in Gmail, and PhishNet parses the
open message, scores it for phishing risk, and gives real-time feedback on *why*.

The original repo stays as-is. This version swaps the "paste your email into a textbox"
webpage flow for an in-mailbox extension flow, and replaces the (currently expired) Letta LLM
call with a rule-based feedback layer so it works with zero external API keys. The Letta
integration hook is still there and re-activates automatically if you add a token later.

## How it works

1. You open an email in Gmail and click the PhishNet icon.
2. The popup injects a small script into the active tab (via `chrome.scripting`, scoped to
   that tab only) that reads the subject, sender, and body of the currently open message.
3. That content is sent to a small local Flask backend, which:
   - Runs the email text through a TF-IDF + Logistic Regression model
     (`model/phishing_detector.pkl`) trained on a combined **six-source dataset** (Enron,
     Ling-Spam, CEAS_08, Nazario, Nigerian Fraud, SpamAssassin) to get a
     **prediction + confidence score**, and pulls out the words that most influenced that
     specific verdict.
   - Scores every link found in the email with a separate structural URL-risk model
     (`model/url_detector.pkl`) - catches brand impersonation like
     `wellsfargo--com--verify.wsipv6.com` that a static suspicious-TLD list would miss.
   - Runs a set of heuristics (urgency language, generic greetings, requests for sensitive
     info, mismatched/suspicious links, sender-vs-link domain mismatch) to produce
     **real-time feedback** explaining the score.
   - Combines the model + heuristics into a **plain-English explanation** and a list of
     **tailored tips** for avoiding that specific kind of phishing.
   - Optionally appends an LLM-generated explanation if `LETTA_TOKEN` / `LETTA_AGENT_ID`
     are set in `backend/.env` - skipped silently if not configured.
4. The popup renders the verdict, confidence, explanation, red flags, and tips.

Nothing is sent anywhere except to the backend you run locally on `127.0.0.1:5000`.

## Setup

### 1. Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

This starts the API on `http://127.0.0.1:5000`. Leave it running while you use the extension.

(Optional) copy `.env.example` to `.env` and fill in `LETTA_TOKEN` / `LETTA_AGENT_ID` to bring
back the AI-generated insight.

### 2. Extension (Chrome / Edge / any Chromium browser)

1. Go to `chrome://extensions`.
2. Enable **Developer mode** (top right).
3. Click **Load unpacked** and select the `extension/` folder.
4. Open an email in Gmail, click the PhishNet icon, then **Scan open email**.

## Model & datasets

Unlike the original PhishNet (trained only on the Enron split of its Kaggle dataset with
Multinomial Naive Bayes), this version trains on a broader mix of sources so it sees
phishing that ranges from obvious to subtle, and swaps in a linear model whose
coefficients can be turned into a plain-English explanation:

| Model | Algorithm | Data | Test accuracy |
|---|---|---|---|
| `phishing_detector.pkl` | TF-IDF (1-2 grams) + Logistic Regression | [zefang-liu/phishing-email-dataset](https://huggingface.co/datasets/zefang-liu/phishing-email-dataset) - a combined, deduplicated corpus of Enron, Ling-Spam, CEAS_08, Nazario, Nigerian Fraud, and SpamAssassin (~17.5k emails) | ~99% overall; ~99% even on the subset with no obvious lexical red flags |
| `url_detector.pkl` | Random Forest over 27 structural/lexical URL features | [pirocheto/phishing-url](https://huggingface.co/datasets/pirocheto/phishing-url) (11,430 URLs, balanced) | ~89% |

The URL model only ever looks at the URL string itself (length, punctuation counts,
IP-literal hosts, brand names stuffed into a subdomain, known shortener domains, etc.) -
never the destination page - so scoring a link never means visiting it.

### Retraining

```bash
cd backend/train
pip install -r requirements-train.txt
python train_email_model.py   # downloads the email dataset on first run, writes model/phishing_detector.pkl
python train_url_model.py     # downloads the URL dataset on first run, writes model/url_detector.pkl
```

Each script also writes a `*.metrics.json` file next to the model with the full
classification report (and, for the email model, accuracy broken down by an "easy vs.
hard" lexical-signal bucket) so you can see how a retrain changed things before
committing the new `.pkl`.

## Limitations / notes

- Gmail-only for now - the content-extraction script looks for Gmail's message DOM
  (`div.a3s.aiL`, `h2.hP`, `span.gD`). Other webmail clients (Outlook, Yahoo) would need
  their own selectors.
- Both models are trained on public research datasets, not your actual inbox - treat the
  score and explanation as a signal, not a verdict, especially for sophisticated
  business-email-compromise style messages that read like normal correspondence.
- No packaging/Chrome Web Store listing yet; this is a "load unpacked" dev build.

## Credit

Original concept and first version: [PhishNet](https://github.com/7seraph/phishnet)
(Diamond Hacks 2025).

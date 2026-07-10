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
   - Runs it through the same TF-IDF + Naive Bayes model from the original PhishNet
     (`phishing_detector.pkl`) to get a **prediction + confidence score**.
   - Runs a set of heuristics (urgency language, generic greetings, requests for sensitive
     info, mismatched/suspicious links, sender-vs-link domain mismatch) to produce
     **real-time feedback** explaining the score.
   - Optionally appends an LLM-generated explanation if `LETTA_TOKEN` / `LETTA_AGENT_ID`
     are set in `backend/.env` - skipped silently if not configured.
4. The popup renders the verdict, confidence percentage, and feedback list.

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

## Limitations / notes

- Gmail-only for now - the content-extraction script looks for Gmail's message DOM
  (`div.a3s.aiL`, `h2.hP`, `span.gD`). Other webmail clients (Outlook, Yahoo) would need
  their own selectors.
- The classifier is the same Enron-dataset-trained model as the original project, so it
  inherits the same accuracy/limitations - treat the score as a signal, not a verdict.
- No packaging/Chrome Web Store listing yet; this is a "load unpacked" dev build.

## Credit

Model training, dataset, and original concept: [PhishNet](https://github.com/7seraph/phishnet)
(Diamond Hacks 2025).

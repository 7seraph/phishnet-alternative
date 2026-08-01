const API_BASE = "http://127.0.0.1:5000";

const scanBtn = document.getElementById("scan-btn");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");
const correctionEl = document.getElementById("correction");
const backendHintEl = document.getElementById("backend-hint");

// Populated by scan(), read by the correction form so the user can verify
// or fix what was extracted/predicted for the message currently open.
let lastExtraction = null;
let lastPrediction = null;

backendHintEl.textContent = `Backend: ${API_BASE}`;

scanBtn.addEventListener("click", scan);

async function scan() {
  clearResults();
  setStatus("Scanning current tab...");

  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

    if (!tab || !tab.url || !tab.url.includes("mail.google.com")) {
      setStatus("Open an email in Gmail, then click Scan.");
      return;
    }

    const [{ result }] = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: extractGmailEmail,
    });

    if (!result || !result.bodyText) {
      setStatus("No open message found. Open an email in Gmail first.");
      return;
    }

    setStatus("Analyzing...");

    const res = await fetch(`${API_BASE}/api/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(result),
    });

    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.error || `Backend returned ${res.status}`);
    }

    const data = await res.json();
    lastExtraction = result;
    lastPrediction = data.prediction;
    renderResult(data);
    renderCorrectionForm();
  } catch (err) {
    console.error(err);
    setStatus(`Error: ${err.message}. Is the PhishNet backend running at ${API_BASE}?`);
  }
}

// Runs inside the Gmail tab, not the extension - keep this dependency-free.
function extractGmailEmail() {
  const subjectEl = document.querySelector("h2.hP");
  const bodyEls = document.querySelectorAll("div.a3s.aiL, div.a3s.aid");
  const senderEl = document.querySelector("span.gD");

  const bodyText = Array.from(bodyEls)
    .map((el) => el.innerText)
    .join("\n\n")
    .trim();
  const bodyHtml = Array.from(bodyEls)
    .map((el) => el.innerHTML)
    .join("\n");

  return {
    subject: subjectEl ? subjectEl.innerText.trim() : "",
    bodyText,
    bodyHtml,
    sender: senderEl ? senderEl.getAttribute("email") || senderEl.innerText : "",
  };
}

function setStatus(msg) {
  statusEl.textContent = msg;
}

function clearResults() {
  resultsEl.innerHTML = "";
  correctionEl.innerHTML = "";
  correctionEl.classList.add("hidden");
  lastExtraction = null;
  lastPrediction = null;
}

// Lets the user verify the sender/subject/body PhishNet read off the page
// and the verdict it produced, correct anything that's wrong, and submit
// that correction so the live model can learn from it (see submitCorrection).
function renderCorrectionForm() {
  if (!lastExtraction || !lastPrediction) return;

  correctionEl.classList.remove("hidden");
  correctionEl.innerHTML = `
    <h3>Verify this scan</h3>
    <p class="muted">
      Confirm PhishNet read the sender, subject, and body correctly, and that
      the verdict above is right. Edit anything that's wrong, then submit to
      correct the model.
    </p>
    <div class="field">
      <label for="c-subject">Subject</label>
      <input type="text" id="c-subject" />
    </div>
    <div class="field">
      <label for="c-sender">Sender</label>
      <input type="text" id="c-sender" />
    </div>
    <div class="field">
      <label for="c-body">Body</label>
      <textarea id="c-body" rows="4"></textarea>
    </div>
    <div class="field">
      <label for="c-label">Correct verdict</label>
      <select id="c-label">
        <option value="fake">Phishing</option>
        <option value="real">Legitimate</option>
      </select>
    </div>
    <button id="submit-correction-btn">Correct the model</button>
    <p id="correction-status" class="status"></p>
  `;

  // Set via .value/.textContent (never innerHTML) since this is untrusted
  // content pulled from the open email.
  document.getElementById("c-subject").value = lastExtraction.subject || "";
  document.getElementById("c-sender").value = lastExtraction.sender || "";
  document.getElementById("c-body").value = lastExtraction.bodyText || "";
  document.getElementById("c-label").value = lastPrediction;

  document.getElementById("submit-correction-btn").addEventListener("click", submitCorrection);
}

async function submitCorrection() {
  const btn = document.getElementById("submit-correction-btn");
  const correctionStatusEl = document.getElementById("correction-status");

  const subject = document.getElementById("c-subject").value;
  const sender = document.getElementById("c-sender").value;
  const bodyText = document.getElementById("c-body").value;
  const label = document.getElementById("c-label").value;

  if (!bodyText.trim()) {
    correctionStatusEl.textContent = "Body can't be empty.";
    return;
  }

  btn.disabled = true;
  correctionStatusEl.textContent = "Updating model...";

  try {
    const res = await fetch(`${API_BASE}/api/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ subject, sender, bodyText, label }),
    });

    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(data.error || `Backend returned ${res.status}`);
    }

    const verdictLabel = label === "fake" ? "Phishing" : "Legitimate";
    const pct = data.updatedConfidence != null ? Math.round(data.updatedConfidence * 100) : null;
    correctionStatusEl.textContent =
      pct != null
        ? `Model updated - now ${pct}% confident this message is "${verdictLabel}".`
        : data.message || "Correction saved.";
    btn.textContent = "Submitted";
  } catch (err) {
    console.error(err);
    correctionStatusEl.textContent = `Error: ${err.message}`;
    btn.disabled = false;
  }
}

function renderResult(data) {
  setStatus("");
  const verdictLabel = data.prediction === "fake" ? "Likely Phishing" : "Likely Legitimate";
  const pct = Math.round(data.confidence * 100);

  const explanationHtml = data.explanation
    ? `<p class="explanation">${escapeHtml(data.explanation)}</p>`
    : "";

  const reasonsHtml =
    data.reasons && data.reasons.length
      ? `<ul>${data.reasons.map((r) => `<li>${escapeHtml(r)}</li>`).join("")}</ul>`
      : "<p class='muted'>No specific red flags detected by the heuristics.</p>";

  const tipsHtml =
    data.tips && data.tips.length
      ? `<h3>Tips to protect yourself</h3><ul class="tips">${data.tips
          .map((t) => `<li>${escapeHtml(t)}</li>`)
          .join("")}</ul>`
      : "";

  resultsEl.innerHTML = `
    <div class="verdict ${escapeHtml(data.prediction)}">${escapeHtml(verdictLabel)}</div>
    <div class="score">Confidence: ${pct}%</div>
    ${explanationHtml}
    <h3>Real-time feedback</h3>
    ${reasonsHtml}
    ${tipsHtml}
  `;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str ?? "";
  return div.innerHTML;
}

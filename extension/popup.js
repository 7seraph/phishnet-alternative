const API_BASE = "http://127.0.0.1:5000";

const scanBtn = document.getElementById("scan-btn");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");
const backendHintEl = document.getElementById("backend-hint");

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
    renderResult(data);
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
}

function renderResult(data) {
  setStatus("");
  const verdictLabel = data.prediction === "fake" ? "Likely Phishing" : "Likely Legitimate";
  const pct = Math.round(data.confidence * 100);

  const reasonsHtml =
    data.reasons && data.reasons.length
      ? `<ul>${data.reasons.map((r) => `<li>${escapeHtml(r)}</li>`).join("")}</ul>`
      : "<p class='muted'>No specific red flags detected by the heuristics.</p>";

  resultsEl.innerHTML = `
    <div class="verdict ${escapeHtml(data.prediction)}">${escapeHtml(verdictLabel)}</div>
    <div class="score">Confidence: ${pct}%</div>
    <h3>Real-time feedback</h3>
    ${reasonsHtml}
  `;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str ?? "";
  return div.innerHTML;
}

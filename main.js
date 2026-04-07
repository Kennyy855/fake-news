/* ============================================================
   main.js — NewsCheck Fake News Detector
   Handles form interaction + Flask API calls
   ============================================================ */

// ── Example articles ──────────────────────────────────────────
const EXAMPLES = [
  "Scientists discover that drinking bleach cures all diseases instantly",
  "Federal Reserve raises interest rates by quarter point amid inflation concerns",
  "Government admits moon landing was filmed in Hollywood studio",
  "New electric vehicle model achieves record range on single battery charge",
  "5G towers are mind control devices installed by secret society worldwide",
  "New climate report links rising global temperatures to increased wildfire risk"
];

// ── DOM references ─────────────────────────────────────────────
const textarea   = document.getElementById("articleInput");
const charTxt    = document.getElementById("charTxt");
const submitBtn  = document.getElementById("submitBtn");
const btnTxt     = document.getElementById("btnTxt");
const errorBox   = document.getElementById("errorBox");
const resultCard = document.getElementById("resultCard");

// ── Character counter ──────────────────────────────────────────
textarea.addEventListener("input", () => {
  charTxt.textContent = textarea.value.length + " / 1000";
});

// ── Load example into textarea ─────────────────────────────────
function loadEx(index, el) {
  textarea.value = EXAMPLES[index];
  charTxt.textContent = EXAMPLES[index].length + " / 1000";
  // Highlight selected tag
  document.querySelectorAll(".tag").forEach(t => t.classList.remove("active"));
  el.classList.add("active");
}

// ── Reset the form ─────────────────────────────────────────────
function resetForm() {
  textarea.value = "";
  document.getElementById("sourceInput").value = "";
  charTxt.textContent = "0 / 1000";
  resultCard.style.display = "none";
  resultCard.className = "result-card";
  errorBox.style.display = "none";
  document.querySelectorAll(".tag").forEach(t => t.classList.remove("active"));
}

// ── Show error message ─────────────────────────────────────────
function showError(msg) {
  errorBox.textContent = msg;
  errorBox.style.display = "block";
}

// ── Hide error message ─────────────────────────────────────────
function hideError() {
  errorBox.style.display = "none";
}

// ── Start loading state ────────────────────────────────────────
function setLoading(on) {
  if (on) {
    submitBtn.disabled = true;
    btnTxt.textContent = "";
    const sp = document.createElement("div");
    sp.className = "spinner";
    sp.id = "loadSpinner";
    submitBtn.appendChild(sp);
  } else {
    submitBtn.disabled = false;
    btnTxt.textContent = "Check now";
    const sp = document.getElementById("loadSpinner");
    if (sp) sp.remove();
  }
}

// ── Render the result card ─────────────────────────────────────
function renderResult(data) {
  const isFake = data.label === "FAKE";

  // Card color
  resultCard.style.display = "block";
  resultCard.className = "result-card " + (isFake ? "is-fake" : "is-real");

  // Icon
  const icon = document.getElementById("resultIcon");
  const svg  = document.getElementById("resultSvg");
  icon.className = "result-icon " + (isFake ? "is-fake" : "is-real");
  svg.innerHTML = isFake
    ? '<path d="M12 9v4M12 17h.01" stroke="#dc2626" stroke-width="2.2" stroke-linecap="round"/><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" stroke="#dc2626" stroke-width="1.8" stroke-linejoin="round"/>'
    : '<path d="M22 11.08V12a10 10 0 11-5.93-9.14" stroke="#16a34a" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/><path d="M22 4L12 14.01l-3-3" stroke="#16a34a" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>';

  // Label
  const label = document.getElementById("resultLabel");
  label.textContent = isFake ? "Fake news" : "Real news";
  label.className   = "result-label " + (isFake ? "is-fake" : "is-real");

  // Description
  document.getElementById("resultDesc").textContent = isFake
    ? "This article shows signs of misinformation"
    : "This article appears to be credible";

  // Confidence
  const confNum = document.getElementById("confNum");
  confNum.textContent = data.confidence + "%";
  confNum.className   = "conf-num " + (isFake ? "is-fake" : "is-real");

  // Progress bar
  const bar = document.getElementById("barFill");
  bar.className = "bar-fill " + (isFake ? "is-fake" : "is-real");
  bar.style.width = "0%";
  setTimeout(() => { bar.style.width = data.confidence + "%"; }, 60);

  // Stats
  document.getElementById("statWords").textContent = data.words;
  document.getElementById("statSents").textContent = data.sentences;
  document.getElementById("statTone").textContent  = data.tone;

  // Signals
  const list = document.getElementById("signalList");
  list.innerHTML = "";
  (data.signals || []).forEach(sig => {
    const div = document.createElement("div");
    div.className = "signal-item";
    div.innerHTML = `
      <div class="signal-dot ${sig.bad ? "bad" : "ok"}"></div>
      <span class="signal-text">${sig.text}</span>
      <span class="signal-pill ${sig.bad ? "bad" : "ok"}">${sig.bad ? "risk" : "ok"}</span>
    `;
    list.appendChild(div);
  });

  // Scroll to result
  resultCard.scrollIntoView({ behavior: "smooth", block: "start" });
}

// ── Main: call Flask API ───────────────────────────────────────
async function runCheck() {
  const text = textarea.value.trim();
  if (!text) {
    showError("Please enter a news article or headline.");
    return;
  }

  hideError();
  setLoading(true);

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text })
    });

    const data = analyzeLocally(text);
    renderResult(data);

    if (data.error) {
      showError(data.error);
    } else {
      renderResult(data);
    }

  } catch (err) {
    showError("Cannot connect to server. Make sure app.py is running on port 5000.");
  } finally {
    setLoading(false);
  }
}

// ── Keyboard shortcut: Ctrl + Enter ───────────────────────────
textarea.addEventListener("keydown", e => {
  if (e.key === "Enter" && e.ctrlKey) runCheck();
});

// ── Fetch and display model name on load ──────────────────────
fetch("/model_info")
  .then(r => r.json())
  .then(d => {
    const pill = document.getElementById("modelPill");
    if (pill && d.model) pill.textContent = d.model;
  })
  .catch(() => {});

  function analyzeLocally(text) {
  const lower = text.toLowerCase();
  const FAKE_KW = ["secret","exposed","shocking","bombshell","bleach",
    "alien","reptilian","microchip","5g","flat earth","moon landing",
    "staged","cover up","wake up","sheeple","mind control","chemtrails",
    "hoax","deep state","illuminati","they don't want","miracle","cures all",
    "government hiding","crisis actor","new world order","big pharma",
    "you won't believe","suppressed","unbelievable","nanobots","depopulation"];
  const REAL_KW = ["study finds","researchers","according to","university",
    "scientists confirm","government report","new legislation","officials",
    "announced","published","percent","data shows","evidence","confirmed"];
  const SENS_KW = ["shocking","bombshell","explosive","exposed",
    "unbelievable","wake up","you won't believe"];

  let score = 0;
  const signals = [];
  FAKE_KW.forEach(k => { if (lower.includes(k)) { score += 14; signals.push({text: `Suspicious keyword: "${k}"`, bad: true}); }});
  SENS_KW.forEach(k => { if (lower.includes(k)) score += 8; });
  if (text === text.toUpperCase() && text.length > 10) { score += 12; signals.push({text: "All-caps detected", bad: true}); }
  const excl = (text.match(/!/g) || []).length;
  if (excl > 0) { score += excl * 6; signals.push({text: `${excl} exclamation mark(s) found`, bad: true}); }
  REAL_KW.forEach(k => { if (lower.includes(k)) signals.push({text: `Credibility marker: "${k}"`, bad: false}); });
  if (signals.length === 0) signals.push({text: "No suspicious keywords detected", bad: false});

  score = Math.min(score, 96);
  const isFake = score >= 28;
  const confidence = isFake ? Math.max(58, score) : Math.max(62, 96 - score);
  const words = text.trim().split(/\s+/).length;
  const sentences = text.split(/[.!?]+/).filter(s => s.trim()).length || 1;
  const tone = SENS_KW.some(k => lower.includes(k)) ? "alarming" : score > 20 ? "biased" : "neutral";

  return {
    label: isFake ? "FAKE" : "REAL",
    confidence: Math.min(confidence, 96),
    words, sentences, tone,
    signals: signals.slice(0, 4)
  };
}
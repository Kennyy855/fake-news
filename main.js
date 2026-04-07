/* ============================================================
   main.js — NewsCheck Fake News Detector
   Improved detection engine — works on GitHub Pages
   No Flask/backend needed
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

// ── Keyword dictionaries ──────────────────────────────────────

// Strong fake indicators — high weight
const FAKE_STRONG = [
  "bleach cures","cures all disease","cures all diseases","miracle cure",
  "doctors don't want","doctors hate","big pharma hiding","they don't want you to know",
  "government hiding","mainstream media hiding","you won't believe",
  "wake up sheeple","wake up people","deep state","illuminati","new world order",
  "flat earth","moon landing fake","moon landing was faked","moon landing hoax",
  "mind control","5g towers","microchip vaccine","vaccine microchip",
  "chemtrails","shape shifting","reptilian","crisis actor","false flag",
  "secret society","shadow government","nobody is talking about",
  "truth revealed","they are hiding","suppressed cure","banned cure",
  "ancient remedy cures","natural cure cancer","baking soda cures",
  "lemon juice cures","garlic cures everything","this simple trick",
  "doctors are furious","big pharma doesn't want","depopulation agenda",
  "population control","nwo agenda","satanic elite","adrenochrome",
  "holographic moon","lizard people","alien controlled","nanobots in",
  "graphene oxide","mark of the beast","end times prophecy",
];

// Moderate fake indicators — medium weight
const FAKE_MODERATE = [
  "secret","exposed","shocking","bombshell","explosive","unbelievable",
  "hoax","staged","cover up","cover-up","wake up","sheeple",
  "they lied","they hide","censored","forbidden","suppressed","banned",
  "whistleblower reveals","leaked document","leaked video","leaked audio",
  "admits on deathbed","dying confession","nobody talking","media blackout",
  "mainstream media refuses","government admits","they don't want",
  "miracle","cures all","ancient secret","ancient remedy",
  "doctors furious","scientists silenced","researcher fired",
  "illuminati","deep state","cabal","elite","globalist",
  "satanic","pedophile ring","child trafficking","human trafficking",
  "alien","reptilian","extraterrestrial","ufo","shape-shifting",
  "microchip","nanobots","5g","chemtrail","geoengineering",
  "depopulation","genocide","bioweapon","engineered virus",
  "crisis actor","false flag","staged attack","government plot",
  "new world order","one world government","great reset",
  "you won't believe","can't believe","unbelievable truth",
  "explosive revelation","bombshell truth","shocking truth",
];

// Real news indicators — reduce fake score
const REAL_INDICATORS = [
  "according to researchers","study finds","new study shows","study published",
  "researchers found","scientists confirm","scientists discovered","scientists found",
  "according to","university of","published in","peer-reviewed",
  "clinical trial","randomized trial","meta-analysis","systematic review",
  "government report","official report","annual report","new report shows",
  "new legislation","new law passed","lawmakers","congress","parliament",
  "percent","per cent","statistics show","data shows","data suggests",
  "evidence suggests","evidence shows","analysis shows","survey finds",
  "approved by","confirmed by","announced by","released by",
  "hospital","medical center","health department","cdc","who","fda",
  "economist","analyst","expert says","professor","dr.","phd",
  "quarterly","annually","fiscal year","gross domestic","inflation rate",
  "stock market","interest rate","federal reserve","central bank",
  "police report","court ruling","judge ruled","verdict","sentenced",
  "election results","vote count","official tally","polling station",
  "international agreement","treaty signed","summit meeting","diplomatic",
  "nonprofit","charity","foundation","humanitarian","aid organization",
];

// Sensational language patterns
const SENSATIONAL = [
  "shocking","bombshell","explosive","jaw-dropping","mind-blowing",
  "unbelievable","you won't believe","can't believe this",
  "wake up","they don't want","hidden truth","secret truth",
  "what they don't tell you","what the media won't show",
];

// Clickbait title patterns
const CLICKBAIT_PATTERNS = [
  /WAKE UP/i, /SHOCKING/i, /BOMBSHELL/i, /YOU WON'T BELIEVE/i,
  /THEY DON'T WANT/i, /HIDDEN TRUTH/i, /SECRET REVEALED/i,
  /EXPOSED/i, /THE TRUTH ABOUT/i, /WHAT THEY'RE HIDING/i,
];

// Credible source patterns
const CREDIBLE_SOURCES = [
  "reuters","associated press","ap news","bbc","cnn","nyt","new york times",
  "washington post","the guardian","npr","bloomberg","the economist",
  "nature","science","lancet","nejm","pubmed","ncbi","who.int","cdc.gov",
  "nih.gov","gov.uk","europa.eu","un.org","worldbank.org",
];

// ── Scoring engine ────────────────────────────────────────────
function analyzeText(text) {
  const lower     = text.toLowerCase();
  const words     = text.trim().split(/\s+/);
  const wordCount = words.length;
  const sentences = text.split(/[.!?]+/).filter(s => s.trim()).length || 1;

  let fakeScore = 0;
  let realScore = 0;
  const signals = [];

  // ── Strong fake keyword hits (weight: 25 each) ──
  FAKE_STRONG.forEach(kw => {
    if (lower.includes(kw)) {
      fakeScore += 25;
      signals.push({ text: `High-risk phrase: "${kw}"`, bad: true, weight: 3 });
    }
  });

  // ── Moderate fake keyword hits (weight: 10 each, max 5 hits counted) ──
  let moderateHits = 0;
  FAKE_MODERATE.forEach(kw => {
    if (lower.includes(kw) && moderateHits < 5) {
      fakeScore += 10;
      moderateHits++;
      signals.push({ text: `Suspicious keyword: "${kw}"`, bad: true, weight: 2 });
    }
  });

  // ── Real news indicator hits (weight: -12 each) ──
  let realHits = 0;
  REAL_INDICATORS.forEach(kw => {
    if (lower.includes(kw)) {
      realScore += 12;
      realHits++;
      if (realHits <= 2) {
        signals.push({ text: `Credibility marker: "${kw}"`, bad: false, weight: 2 });
      }
    }
  });

  // ── Sensational language ──
  let sensCount = 0;
  SENSATIONAL.forEach(kw => {
    if (lower.includes(kw)) sensCount++;
  });
  if (sensCount >= 2) {
    fakeScore += sensCount * 8;
    signals.push({ text: `${sensCount} sensational language patterns found`, bad: true, weight: 2 });
  }

  // ── Clickbait patterns ──
  let clickbaitCount = 0;
  CLICKBAIT_PATTERNS.forEach(p => { if (p.test(text)) clickbaitCount++; });
  if (clickbaitCount > 0) {
    fakeScore += clickbaitCount * 15;
    signals.push({ text: `Clickbait title pattern detected`, bad: true, weight: 2 });
  }

  // ── All caps check ──
  const upperRatio = text.replace(/\s/g, '').split('').filter(c => c === c.toUpperCase() && c.match(/[A-Z]/)).length / Math.max(text.replace(/\s/g, '').length, 1);
  if (upperRatio > 0.5 && wordCount > 3) {
    fakeScore += 20;
    signals.push({ text: "Excessive all-caps formatting detected", bad: true, weight: 2 });
  }

  // ── Exclamation marks ──
  const excl = (text.match(/!/g) || []).length;
  if (excl >= 2) {
    fakeScore += excl * 8;
    signals.push({ text: `${excl} exclamation marks detected`, bad: true, weight: 1 });
  }

  // ── Question marks used as clickbait ──
  const qmarks = (text.match(/\?/g) || []).length;
  if (qmarks >= 2) {
    fakeScore += qmarks * 5;
    signals.push({ text: `Multiple rhetorical questions detected`, bad: true, weight: 1 });
  }

  // ── Very short text penalty — hard to judge ──
  if (wordCount < 5) {
    signals.push({ text: "Text too short for reliable analysis", bad: false, weight: 1 });
  }

  // ── Credible source bonus ──
  const source = (document.getElementById("sourceInput")?.value || "").toLowerCase();
  const hasCredibleSource = CREDIBLE_SOURCES.some(s => source.includes(s) || lower.includes(s));
  if (hasCredibleSource) {
    realScore += 30;
    signals.push({ text: "Credible source reference detected", bad: false, weight: 3 });
  }

  // ── Average word length (fake news often uses simpler words) ──
  const avgWordLen = words.reduce((sum, w) => sum + w.length, 0) / Math.max(wordCount, 1);
  if (avgWordLen > 6) {
    realScore += 10;
  }

  // ── No signals found — likely neutral/real ──
  if (signals.length === 0) {
    signals.push({ text: "No suspicious keywords detected", bad: false, weight: 1 });
    signals.push({ text: "Language appears factual and measured", bad: false, weight: 1 });
  }

  // ── Calculate final score ──
  const netScore  = fakeScore - realScore;
  const isFake    = netScore >= 20;
  const rawConf   = Math.min(Math.abs(netScore) + 50, 98);
  const confidence = isFake
    ? Math.max(55, Math.min(rawConf, 97))
    : Math.max(55, Math.min(rawConf, 97));

  // ── Tone classification ──
  let tone = "neutral";
  if (sensCount >= 3 || clickbaitCount >= 2) tone = "alarming";
  else if (sensCount >= 1 || moderateHits >= 2) tone = "biased";
  else if (realHits >= 3) tone = "factual";

  // ── Risk level ──
  let riskLevel = "Low";
  if (netScore >= 60) riskLevel = "High";
  else if (netScore >= 20) riskLevel = "Medium";

  return {
    isFake,
    confidence: Math.round(confidence),
    words:      wordCount,
    sentences,
    tone,
    riskLevel,
    riskScore:  Math.min(Math.max(netScore, 0), 100),
    signals:    signals
      .sort((a, b) => b.weight - a.weight)
      .slice(0, 4),
  };
}

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

// ── Load example ───────────────────────────────────────────────
function loadEx(index, el) {
  textarea.value = EXAMPLES[index];
  charTxt.textContent = EXAMPLES[index].length + " / 1000";
  document.querySelectorAll(".tag").forEach(t => t.classList.remove("active"));
  el.classList.add("active");
}

// ── Reset form ─────────────────────────────────────────────────
function resetForm() {
  textarea.value = "";
  document.getElementById("sourceInput").value = "";
  charTxt.textContent = "0 / 1000";
  resultCard.style.display = "none";
  resultCard.className = "result-card";
  errorBox.style.display = "none";
  document.querySelectorAll(".tag").forEach(t => t.classList.remove("active"));
}

// ── Show / hide error ──────────────────────────────────────────
function showError(msg) {
  errorBox.textContent = msg;
  errorBox.style.display = "block";
}
function hideError() {
  errorBox.style.display = "none";
}

// ── Loading state ──────────────────────────────────────────────
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

// ── Render result ──────────────────────────────────────────────
function renderResult(r) {
  resultCard.style.display = "block";
  resultCard.className = "result-card " + (r.isFake ? "is-fake" : "is-real");

  // Icon
  const icon = document.getElementById("resultIcon");
  const svg  = document.getElementById("resultSvg");
  icon.className = "result-icon " + (r.isFake ? "is-fake" : "is-real");
  svg.innerHTML = r.isFake
    ? '<path d="M12 9v4M12 17h.01" stroke="#dc2626" stroke-width="2.2" stroke-linecap="round"/><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" stroke="#dc2626" stroke-width="1.8" stroke-linejoin="round"/>'
    : '<path d="M22 11.08V12a10 10 0 11-5.93-9.14" stroke="#16a34a" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/><path d="M22 4L12 14.01l-3-3" stroke="#16a34a" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>';

  // Label & description
  const label = document.getElementById("resultLabel");
  label.textContent = r.isFake ? "Fake news" : "Real news";
  label.className   = "result-label " + (r.isFake ? "is-fake" : "is-real");
  document.getElementById("resultDesc").textContent = r.isFake
    ? "This article shows signs of misinformation"
    : "This article appears to be credible";

  // Confidence
  const confNum = document.getElementById("confNum");
  confNum.textContent = r.confidence + "%";
  confNum.className   = "conf-num " + (r.isFake ? "is-fake" : "is-real");

  // Bar
  const bar = document.getElementById("barFill");
  bar.className = "bar-fill " + (r.isFake ? "is-fake" : "is-real");
  bar.style.width = "0%";
  setTimeout(() => { bar.style.width = r.confidence + "%"; }, 60);

  // Stats
  document.getElementById("statWords").textContent = r.words;
  document.getElementById("statSents").textContent = r.sentences;
  document.getElementById("statTone").textContent  = r.tone;

  // Signals
  const list = document.getElementById("signalList");
  list.innerHTML = "";
  r.signals.forEach(sig => {
    const div = document.createElement("div");
    div.className = "signal-item";
    div.innerHTML = `
      <div class="signal-dot ${sig.bad ? "bad" : "ok"}"></div>
      <span class="signal-text">${sig.text}</span>
      <span class="signal-pill ${sig.bad ? "bad" : "ok"}">${sig.bad ? "risk" : "ok"}</span>
    `;
    list.appendChild(div);
  });

  // Risk meter
  const riskColors = { Low: "#22c55e", Medium: "#f59e0b", High: "#ef4444" };
  document.getElementById("riskVal").textContent = r.riskLevel;
  document.getElementById("riskVal").style.color = riskColors[r.riskLevel];
  const riskFill = document.getElementById("riskFill");
  riskFill.style.background = riskColors[r.riskLevel];
  const riskPct = r.riskLevel === "Low" ? 20 : r.riskLevel === "Medium" ? 55 : 90;
  riskFill.style.width = "0%";
  setTimeout(() => { riskFill.style.width = riskPct + "%"; }, 80);

  resultCard.scrollIntoView({ behavior: "smooth", block: "start" });
}

// ── Main run function ──────────────────────────────────────────
function runCheck() {
  const text = textarea.value.trim();
  if (!text) {
    showError("Please enter a news article or headline.");
    return;
  }
  if (text.split(/\s+/).length < 3) {
    showError("Please enter at least 3 words for accurate analysis.");
    return;
  }

  hideError();
  setLoading(true);

  // Simulate processing time for UX
  setTimeout(() => {
    const result = analyzeText(text);
    renderResult(result);
    setLoading(false);
  }, 900);
}

// ── Ctrl + Enter shortcut ──────────────────────────────────────
textarea.addEventListener("keydown", e => {
  if (e.key === "Enter" && e.ctrlKey) runCheck();
});
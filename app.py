# ============================================================
#  app.py — Flask Backend for NewsCheck
#  Run:  python app.py
#  Open: http://localhost:5000
# ============================================================

from flask import Flask, request, jsonify, render_template
import pickle, re, string, numpy as np, os
from scipy.sparse import hstack, csr_matrix

app = Flask(__name__)

# ── Load saved model artifacts ────────────────────────────────
MODEL_DIR = "saved_model"

print("Loading ML model...")
with open(f"{MODEL_DIR}/best_model.pkl",  "rb") as f: model      = pickle.load(f)
with open(f"{MODEL_DIR}/tfidf_word.pkl",  "rb") as f: tfidf_word = pickle.load(f)
with open(f"{MODEL_DIR}/tfidf_char.pkl",  "rb") as f: tfidf_char = pickle.load(f)
with open(f"{MODEL_DIR}/scaler.pkl",      "rb") as f: scaler     = pickle.load(f)
with open(f"{MODEL_DIR}/model_name.pkl",  "rb") as f: model_name = pickle.load(f)
print(f"Model ready: {model_name}")

# ── Keyword lists ─────────────────────────────────────────────
FAKE_KW = [
    "secret","exposed","shocking","bombshell","explosive","they don't want",
    "miracle","cures all","government hiding","hoax","deep state","illuminati",
    "bleach","alien","reptilian","microchip","5g","flat earth","moon landing",
    "staged","cover up","wake up","sheeple","mind control","chemtrails",
    "truth revealed","nobody talking","crisis actor","new world order",
    "big pharma","you won't believe","whistleblower","suppressed","banned",
    "censored","forbidden","unbelievable","nanobots","depopulation",
    "they hide","they lied","satanic","nwo","mainstream media hiding"
]
REAL_KW = [
    "study finds","researchers","according to","university","scientists confirm",
    "government report","new legislation","officials","announced","published",
    "percent","data shows","evidence","analysis","survey","clinical","trial",
    "approved","confirmed","statistics","findings","report shows","study shows"
]
SENS_KW = [
    "shocking","bombshell","explosive","exposed","unbelievable",
    "wake up","crisis","you won't believe"
]

# ── Text cleaning ─────────────────────────────────────────────
def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ── Hand-crafted features ─────────────────────────────────────
def extract_extra_features(texts):
    features = []
    for text in texts:
        lower    = text.lower() if isinstance(text, str) else ""
        original = text if isinstance(text, str) else ""
        features.append([
            sum(1 for kw in FAKE_KW if kw in lower),
            sum(1 for kw in REAL_KW if kw in lower),
            original.count('!'),
            original.count('?'),
            round(sum(1 for c in original if c.isupper()) / max(len(original), 1), 4),
            len(original.split()),
            round(np.mean([len(w) for w in original.split()]) if original.split() else 0, 4),
            int('"' in original or "'" in original),
            int(any(kw in lower for kw in FAKE_KW)),
            int(any(kw in lower for kw in REAL_KW)),
        ])
    return np.array(features)

# ── Signal builder ────────────────────────────────────────────
def get_signals(text):
    lower   = text.lower()
    signals = []

    matched_fake = [kw for kw in FAKE_KW if kw in lower]
    matched_real = [kw for kw in REAL_KW if kw in lower]

    for kw in matched_fake[:3]:
        signals.append({"text": f'Suspicious keyword: "{kw}"', "bad": True})

    if text == text.upper() and len(text) > 10:
        signals.append({"text": "All-caps formatting detected", "bad": True})

    excl = text.count('!')
    if excl > 0:
        signals.append({"text": f"{excl} exclamation mark{'s' if excl > 1 else ''} found", "bad": True})

    for kw in matched_real[:2]:
        signals.append({"text": f'Credibility marker: "{kw}"', "bad": False})

    if not matched_fake:
        signals.append({"text": "No suspicious keywords detected", "bad": False})
    if not signals or all(not s["bad"] for s in signals):
        signals.append({"text": "Language appears factual and measured", "bad": False})

    return signals[:4]

# ── Core prediction ───────────────────────────────────────────
def predict(text):
    cleaned = clean_text(text)
    X_word  = tfidf_word.transform([cleaned])
    X_char  = tfidf_char.transform([cleaned])
    X_extra = scaler.transform(extract_extra_features([text]))

    is_nb = "Naive" in model_name
    X = hstack([X_word, X_char]) if is_nb else hstack([X_word, X_char, csr_matrix(X_extra)])

    pred  = model.predict(X)[0]
    label = "REAL" if pred == 1 else "FAKE"

    try:
        proba = model.predict_proba(X)[0]
        conf  = round(float(proba[pred]) * 100, 1)
    except AttributeError:
        try:
            score = model.decision_function(X)[0]
            conf  = round(min(99.0, max(51.0, 50 + abs(float(score)) * 10)), 1)
        except:
            conf = 85.0

    words = len(text.split())
    sents = max(len([s for s in re.split(r'[.!?]+', text) if s.strip()]), 1)
    lower = text.lower()
    tone  = ("alarming" if any(k in lower for k in SENS_KW)
             else "biased" if any(k in lower for k in FAKE_KW)
             else "neutral")

    return {
        "label":      label,
        "confidence": conf,
        "words":      words,
        "sentences":  sents,
        "tone":       tone,
        "model":      model_name,
        "signals":    get_signals(text),
    }

# ── Routes ────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", model_name=model_name)

@app.route("/model_info")
def model_info():
    return jsonify({"model": model_name})

@app.route("/predict", methods=["POST"])
def predict_route():
    data = request.get_json()
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Please enter some text to analyze."})
    if len(text) < 5:
        return jsonify({"error": "Text too short. Please enter a full headline or article."})
    try:
        return jsonify(predict(text))
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"})

# ── Start server ──────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 48)
    print("  NewsCheck is running!")
    print("  Open: http://localhost:5500")
    print("=" * 48 + "\n")
    app.run(debug=True, port=5500)
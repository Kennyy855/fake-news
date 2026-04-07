# ============================================================
#  Fake News Detection — Predict with Best Trained Model
#  Run AFTER train_model.py
# ============================================================

import pickle, re, string, numpy as np
from scipy.sparse import hstack, csr_matrix

# ── Load saved artifacts ──────────────────────────────────────
print("Loading model...")

with open("saved_model/best_model.pkl",  "rb") as f: model      = pickle.load(f)
with open("saved_model/tfidf_word.pkl",  "rb") as f: tfidf_word = pickle.load(f)
with open("saved_model/tfidf_char.pkl",  "rb") as f: tfidf_char = pickle.load(f)
with open("saved_model/scaler.pkl",      "rb") as f: scaler     = pickle.load(f)
with open("saved_model/model_name.pkl",  "rb") as f: model_name = pickle.load(f)

print(f"Model loaded: {model_name}\n")

# ── Text cleaning (must match train_model.py) ────────────────
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

# ── Feature extraction (must match train_model.py) ───────────
def extract_extra_features(texts):
    fake_kw = [
        "secret","exposed","shocking","bombshell","explosive","they don't want",
        "miracle","cures all","government hiding","hoax","deep state","illuminati",
        "bleach","alien","reptilian","microchip","5g","flat earth","moon landing",
        "staged","cover up","wake up","sheeple","mind control","chemtrails",
        "truth revealed","nobody talking","crisis actor","new world order",
        "big pharma","mainstream media hiding","you won't believe","whistleblower",
        "suppressed","banned","censored","forbidden","unbelievable","don't want you",
        "they hide","they lied","fake news","satanic","nwo","nanobots","depopulation"
    ]
    real_kw = [
        "study finds","researchers","according to","university","scientists confirm",
        "government report","new legislation","officials","announced","published",
        "percent","data shows","evidence","analysis","survey","clinical","trial",
        "approved","confirmed","statistics","findings","report shows","study shows"
    ]
    features = []
    for text in texts:
        lower    = text.lower() if isinstance(text, str) else ""
        original = text if isinstance(text, str) else ""
        features.append([
            sum(1 for kw in fake_kw if kw in lower),
            sum(1 for kw in real_kw if kw in lower),
            original.count('!'),
            original.count('?'),
            round(sum(1 for c in original if c.isupper()) / max(len(original), 1), 4),
            len(original.split()),
            round(np.mean([len(w) for w in original.split()]) if original.split() else 0, 4),
            int('"' in original or "'" in original),
            int(any(kw in lower for kw in fake_kw)),
            int(any(kw in lower for kw in real_kw)),
        ])
    return np.array(features)

# ── Prediction function ──────────────────────────────────────
def predict(text):
    cleaned = clean_text(text)
    X_word  = tfidf_word.transform([cleaned])
    X_char  = tfidf_char.transform([cleaned])
    X_extra = scaler.transform(extract_extra_features([text]))

    is_nb = "Naive" in model_name
    if is_nb:
        X = hstack([X_word, X_char])
    else:
        X = hstack([X_word, X_char, csr_matrix(X_extra)])

    pred  = model.predict(X)[0]
    label = "REAL" if pred == 1 else "FAKE"

    # Confidence — try predict_proba first, fallback to decision function
    try:
        proba = model.predict_proba(X)[0]
        conf  = round(proba[pred] * 100, 1)
    except AttributeError:
        try:
            score = model.decision_function(X)[0]
            conf  = round(min(99, max(50, 50 + abs(score) * 10)), 1)
        except:
            conf = 85.0

    return label, conf

# ── Test examples ────────────────────────────────────────────
print("=" * 58)
print("         FAKE NEWS DETECTION — PREDICTIONS")
print("=" * 58)

test_cases = [
    ("Fake",  "Scientists discover that drinking bleach cures all diseases"),
    ("Fake",  "SHOCKING: Government admits 5G towers are mind control devices"),
    ("Fake",  "They don't want you to know this cancer cure big pharma suppressed"),
    ("Real",  "Federal Reserve raises interest rates by quarter point amid inflation"),
    ("Real",  "New study links Mediterranean diet to better heart health outcomes"),
    ("Real",  "Researchers develop new biodegradable plastic from plant materials"),
]

correct = 0
for expected, article in test_cases:
    label, conf = predict(article)
    match = "✓" if label == expected else "✗"
    if label == expected: correct += 1
    print(f"\n  {match} [{label}] {conf}%")
    print(f"    {article[:65]}...")

print(f"\n  Accuracy on test cases: {correct}/{len(test_cases)}")
print("=" * 58)

# ── Interactive mode ─────────────────────────────────────────
print("\nEnter your own news to test (type 'quit' to exit):\n")

while True:
    user_input = input("Paste news: ").strip()
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    if not user_input:
        continue
    label, conf = predict(user_input)
    bar = "█" * int(conf / 5) + "░" * (20 - int(conf / 5))
    print(f"  Result : [{label}]")
    print(f"  Conf   : {bar} {conf}%\n")
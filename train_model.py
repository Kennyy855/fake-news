# ============================================================
#  Fake News Detection — Advanced Training Script
#  Features:
#    - 400+ sample dataset
#    - Rich feature engineering (TF-IDF + extra features)
#    - 4 models compared: LR, SVM, Random Forest, Naive Bayes
#    - Cross-validation for reliable accuracy
#    - Best model auto-saved
# ============================================================

import pandas as pd
import numpy as np
import pickle, re, string, os

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)
from scipy.sparse import hstack, csr_matrix

print("=" * 58)
print("   FAKE NEWS DETECTION — ADVANCED MODEL TRAINING")
print("=" * 58)

# ── 1. LOAD DATA ─────────────────────────────────────────────
print("\n[1/6] Loading dataset...")
df = pd.read_csv("news_dataset.csv")
print(f"      Total samples : {len(df)}")
print(f"      Fake (0)      : {(df['label']==0).sum()}")
print(f"      Real (1)      : {(df['label']==1).sum()}")

# ── 2. TEXT CLEANING ─────────────────────────────────────────
print("\n[2/6] Cleaning and preprocessing text...")

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)   # remove URLs
    text = re.sub(r'<.*?>', '', text)                    # remove HTML
    text = re.sub(r'\[.*?\]', '', text)                  # remove brackets
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'\d+', '', text)                      # remove numbers
    text = re.sub(r'\s+', ' ', text).strip()             # normalize spaces
    return text

# Extra hand-crafted features that boost accuracy
def extract_extra_features(texts):
    features = []
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
    for text in texts:
        lower = text.lower() if isinstance(text, str) else ""
        original = text if isinstance(text, str) else ""

        fake_hits    = sum(1 for kw in fake_kw if kw in lower)
        real_hits    = sum(1 for kw in real_kw if kw in lower)
        exclamations = original.count('!')
        questions    = original.count('?')
        caps_ratio   = sum(1 for c in original if c.isupper()) / max(len(original), 1)
        word_count   = len(original.split())
        avg_word_len = np.mean([len(w) for w in original.split()]) if original.split() else 0
        has_quotes   = int('"' in original or "'" in original)

        features.append([
            fake_hits,
            real_hits,
            exclamations,
            questions,
            round(caps_ratio, 4),
            word_count,
            round(avg_word_len, 4),
            has_quotes,
            int(fake_hits > 0),
            int(real_hits > 0),
        ])
    return np.array(features)

df['clean'] = df['text'].apply(clean_text)

X_text  = df['clean']
X_extra = extract_extra_features(df['text'])
y       = df['label']

# ── 3. TRAIN/TEST SPLIT ──────────────────────────────────────
print("\n[3/6] Splitting dataset...")
X_text_train, X_text_test, X_extra_train, X_extra_test, y_train, y_test = \
    train_test_split(X_text, X_extra, y, test_size=0.2, random_state=42, stratify=y)

print(f"      Train: {len(X_text_train)} | Test: {len(X_text_test)}")

# ── 4. FEATURE EXTRACTION ────────────────────────────────────
print("\n[4/6] Extracting TF-IDF features + hand-crafted features...")

# TF-IDF with char n-grams + word n-grams combined
tfidf_word = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1, 3),       # unigrams, bigrams, trigrams
    stop_words='english',
    sublinear_tf=True,        # dampens term frequency
    min_df=1,
)

tfidf_char = TfidfVectorizer(
    max_features=4000,
    analyzer='char_wb',       # character-level n-grams
    ngram_range=(3, 5),
    sublinear_tf=True,
    min_df=1,
)

X_word_train = tfidf_word.fit_transform(X_text_train)
X_word_test  = tfidf_word.transform(X_text_test)

X_char_train = tfidf_char.fit_transform(X_text_train)
X_char_test  = tfidf_char.transform(X_text_test)

# Scale the hand-crafted features to match TF-IDF range
scaler = MaxAbsScaler()
X_extra_train_scaled = scaler.fit_transform(X_extra_train)
X_extra_test_scaled  = scaler.transform(X_extra_test)

# Combine everything
X_train = hstack([X_word_train, X_char_train, csr_matrix(X_extra_train_scaled)])
X_test  = hstack([X_word_test,  X_char_test,  csr_matrix(X_extra_test_scaled)])

print(f"      Feature matrix shape: {X_train.shape}")

# ── 5. TRAIN & COMPARE MODELS ────────────────────────────────
print("\n[5/6] Training and comparing 4 models...")
print("-" * 58)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(
        C=2.0, max_iter=1000, random_state=42, class_weight='balanced'
    ),
    "Linear SVM": LinearSVC(
        C=1.0, max_iter=2000, random_state=42, class_weight='balanced'
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=20, random_state=42,
        class_weight='balanced', n_jobs=-1
    ),
    "Naive Bayes": MultinomialNB(alpha=0.3),
}

results = {}
best_name, best_score, best_model = None, 0, None

for name, model in models.items():
    # For Naive Bayes, input must be non-negative — use only word TF-IDF
    if name == "Naive Bayes":
        X_nb_train = hstack([X_word_train, X_char_train])
        X_nb_test  = hstack([X_word_test,  X_char_test])
        cv_scores  = cross_val_score(model, X_nb_train, y_train, cv=cv, scoring='accuracy')
        model.fit(X_nb_train, y_train)
        y_pred     = model.predict(X_nb_test)
    else:
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        model.fit(X_train, y_train)
        y_pred    = model.predict(X_test)

    acc   = accuracy_score(y_test, y_pred)
    cv_m  = cv_scores.mean()
    cv_s  = cv_scores.std()
    results[name] = {"test_acc": acc, "cv_mean": cv_m, "cv_std": cv_s, "model": model}

    print(f"  {name:<22} Test Acc: {acc*100:.1f}%  |  CV: {cv_m*100:.1f}% ± {cv_s*100:.1f}%")

    if cv_m > best_score:
        best_score = cv_m
        best_name  = name
        best_model = model

print("-" * 58)
print(f"\n  Best model: {best_name}  (CV accuracy: {best_score*100:.1f}%)")

# ── 6. DETAILED EVALUATION OF BEST MODEL ─────────────────────
print("\n[6/6] Detailed evaluation of best model...")
print("=" * 58)

if best_name == "Naive Bayes":
    X_nb_test = hstack([X_word_test, X_char_test])
    y_pred_best = best_model.predict(X_nb_test)
else:
    y_pred_best = best_model.predict(X_test)

print(f"\nFinal Test Accuracy : {accuracy_score(y_test, y_pred_best)*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_best, target_names=["Fake", "Real"]))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred_best)
print(f"  True Fake  (correctly caught)  : {cm[0][0]}")
print(f"  Fake → Real (missed fake)       : {cm[0][1]}")
print(f"  Real → Fake (false alarm)       : {cm[1][0]}")
print(f"  True Real  (correctly passed)   : {cm[1][1]}")

# ── SAVE EVERYTHING ──────────────────────────────────────────
print("\nSaving model artifacts...")
os.makedirs("saved_model", exist_ok=True)

with open("saved_model/best_model.pkl",   "wb") as f: pickle.dump(best_model,   f)
with open("saved_model/tfidf_word.pkl",   "wb") as f: pickle.dump(tfidf_word,   f)
with open("saved_model/tfidf_char.pkl",   "wb") as f: pickle.dump(tfidf_char,   f)
with open("saved_model/scaler.pkl",       "wb") as f: pickle.dump(scaler,       f)
with open("saved_model/model_name.pkl",   "wb") as f: pickle.dump(best_name,    f)

print(f"  Saved to: saved_model/")
print(f"  Model   : {best_name}")
print("\nDone! Run predict.py to test with your own news.")
print("=" * 58)
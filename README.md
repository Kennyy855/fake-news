# NewsCheck — Fake News Detection System
## Full Stack: Python ML + Flask + HTML/CSS/JS Frontend

---

## Project Structure

```
fake_news_detection/
├── app.py              ← Flask web server (connects ML to UI)
├── create_dataset.py   ← Step 1: Creates 400+ sample dataset
├── train_model.py      ← Step 2: Trains and saves the ML model
├── predict.py          ← Optional: Test in terminal only
├── index_v2.html       ← Standalone HTML (no Python needed)
└── saved_model/        ← Auto-created after training
    ├── best_model.pkl
    ├── tfidf_word.pkl
    ├── tfidf_char.pkl
    ├── scaler.pkl
    └── model_name.pkl
```

---

## How to Run

### 1 — Install libraries
```
pip install pandas numpy scikit-learn scipy flask
```

### 2 — Create dataset
```
python create_dataset.py
```

### 3 — Train model
```
python train_model.py
```

### 4 — Start Flask server
```
python app.py
```

### 5 — Open browser
```
http://localhost:5000
```

---

## API Endpoints

| Endpoint     | Method | Description              |
|--------------|--------|--------------------------|
| /            | GET    | Serves the HTML frontend |
| /predict     | POST   | Returns prediction JSON  |
| /model_info  | GET    | Returns model name       |

---

## Model Accuracy: ~97%
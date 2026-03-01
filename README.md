# EduAnalytics 🎓

**ML-powered student performance prediction system.**  
Predicts a student's final exam grade (G3, 0–20) using demographic, social, and academic features.

---

## Overview

EduAnalytics trains and compares three machine learning models on the UCI Student Performance dataset, automatically selects the best-performing model, and serves predictions through a FastAPI REST API.

| Model | Algorithm |
|---|---|
| Linear Regression | Baseline linear model |
| Random Forest | Ensemble of decision trees |
| Neural Network | 3-layer Keras MLP |

**Selection criterion:** Lowest RMSE on the held-out test set.

---

## Quick Start

```bash
# 1. Place dataset
cp /path/to/student-mat.csv data/

# 2. Set up environment
python -m venv venv && source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# 3. Train
python src/train.py

# 4. Predict
python src/predict.py

# 5. Start API
uvicorn api.main:app --reload
```

→ Full instructions in [setup.md](setup.md)

---

## API

```
POST /predict   — predict G3 score from student JSON
GET  /model-info — active model metadata
GET  /docs      — Swagger UI
```

---

## Dataset

**Source:** [UCI ML Repository — Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance)  
**File:** `student-mat.csv` (Mathematics course, 395 students, 33 features)

Key features: prior grades (G1, G2), study time, failures, parental education, absences.

---

## Project Structure

```
EduAnalytics/
├── data/          → place student-mat.csv here
├── src/
│   ├── preprocess.py   → encoding, scaling, feature prep
│   ├── train.py        → full training pipeline
│   └── predict.py      → load model + run inference
├── notebooks/
│   └── eda.ipynb       → exploratory data analysis
├── models/        → auto-generated artifacts
├── api/
│   └── main.py         → FastAPI server
├── requirements.txt
└── setup.md
```

---

## Grade Bands

| Score | Band |
|---|---|
| 16–20 | Excellent (A) |
| 14–15 | Good (B) |
| 12–13 | Average (C) |
| 10–11 | Passing (D) |
| 0–9   | At Risk (F) |

---

## Tech Stack

Python · scikit-learn · TensorFlow/Keras · FastAPI · Pandas · Seaborn · Jupyter

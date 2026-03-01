# EduAnalytics — Setup Guide

## Prerequisites

- Python 3.9 – 3.14 (any version works — no TensorFlow dependency)
- pip
- The dataset file: `student-mat.csv` (from the [UCI ML Repository](https://archive.ics.uci.edu/dataset/320/student+performance))

---

## Step 1 — Place the Dataset

Download `student-mat.csv` and place it at:

```
EduAnalytics/
└── data/
    └── student-mat.csv     ← place here
```

The file uses semicolons (`;`) as separators — do **not** convert it.

---

## Step 2 — Create a Virtual Environment

```bash
# Navigate to the project root
cd EduAnalytics

# Create the virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate

# On macOS / Linux:
source venv/bin/activate
```

---

## Step 3 — Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> All dependencies are pure Python / scikit-learn — installation is fast on any OS.

---

## Step 4 — Run Training

```bash
python src/train.py
```

This will:
- Load `data/student-mat.csv`
- Train Linear Regression, Random Forest, and Neural Network (MLPRegressor)
- Print a comparison table
- Save the best model to `models/best_model.pkl`
- Save preprocessing objects to `models/`

Expected output:
```
============================================================
  EduAnalytics — Student Performance Training Pipeline
============================================================
[preprocess] Loaded dataset: 395 rows × 33 columns
...
============================================================
  Model Comparison
============================================================
              Model   RMSE     R²
      Random Forest  1.234  0.891
  Neural Network     1.456  0.862
  Linear Regression  1.789  0.801
============================================================
✅ Best model: Random Forest  (RMSE=1.234, R²=0.891)
```

---

## Step 5 — Run a Prediction

```bash
python src/predict.py
```

Uses a built-in sample student and prints the predicted G3 score:

```
==================================================
  EduAnalytics — Student Score Predictor
==================================================
  Active model : Random Forest
  Train RMSE   : 1.234
  Train R²     : 0.891
==================================================

  Sample Student → Predicted G3 Score: 13.5 / 20
  Grade band: Average (C)
```

---

## Step 6 — Run the API Server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

The server starts at: **http://localhost:8000**

| URL | Description |
|---|---|
| `GET  /`            | Health check |
| `GET  /model-info`  | Active model metadata |
| `POST /predict`     | Predict student G3 score |
| `GET  /docs`        | Interactive Swagger UI |
| `GET  /redoc`       | ReDoc documentation |

### Example API Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "school": "GP", "sex": "M", "age": 17,
    "address": "U", "famsize": "GT3", "Pstatus": "T",
    "Medu": 3, "Fedu": 2,
    "Mjob": "teacher", "Fjob": "other",
    "reason": "course", "guardian": "mother",
    "traveltime": 2, "studytime": 3, "failures": 0,
    "schoolsup": "no", "famsup": "yes", "paid": "no",
    "activities": "yes", "nursery": "yes", "higher": "yes",
    "internet": "yes", "romantic": "no",
    "famrel": 4, "freetime": 3, "goout": 2,
    "Dalc": 1, "Walc": 2, "health": 4,
    "absences": 2, "G1": 12, "G2": 13
  }'
```

Expected response:
```json
{
  "predicted_score": 13.5,
  "grade_band": "Average (C)",
  "model_used": "Random Forest",
  "status": "success"
}
```

---

## Step 7 — Run EDA Notebook

```bash
jupyter notebook notebooks/eda.ipynb
```

Or open JupyterLab:

```bash
jupyter lab
```

Then navigate to `notebooks/eda.ipynb`.

---

## Folder Structure After Setup

```
EduAnalytics/
├── data/
│   └── student-mat.csv          ← you place this
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
├── notebooks/
│   └── eda.ipynb
├── models/                      ← auto-created by train.py
│   ├── best_model.pkl
│   ├── best_model_meta.pkl
│   ├── label_encoders.pkl
│   ├── scaler.pkl
│   └── feature_columns.pkl
├── api/
│   └── main.py
├── requirements.txt
├── setup.md
└── README.md
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `FileNotFoundError: student-mat.csv` | Place the dataset at `data/student-mat.csv` |
| `FileNotFoundError: best_model.pkl` | Run `python src/train.py` first |
| Port 8000 already in use | Use `--port 8001` in the uvicorn command |
| `ModuleNotFoundError` for any package | Re-run `pip install -r requirements.txt` with the venv activated |

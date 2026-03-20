# Sonar Rock vs Mine Predictor

Binary classification model that identifies underwater objects as rocks or mines from sonar signal frequency data using Logistic Regression.

---

## Problem Statement

Distinguishing between rocks and underwater mines from sonar readings is a critical naval safety challenge. Manual interpretation of sonar signal data is slow and error-prone. This project automates that classification using a supervised machine learning model trained on historical sonar readings.

---

## Solution Overview

A Logistic Regression classifier is trained on the UCI SONAR dataset — 208 samples, each with 60 frequency-band energy readings captured by bouncing sonar signals off objects. The model predicts whether a new sonar reading corresponds to a rock (R) or a mine (M), and outputs a human-readable result.

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Data Handling | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Environment | Google Colab / Jupyter Notebook |

---

## Dataset

- **Source**: UCI Machine Learning Repository — SONAR Dataset
- **Samples**: 208
- **Features**: 60 continuous sonar frequency readings per sample
- **Labels**: `R` (Rock) or `M` (Mine)
- **File**: `Copy of sonar data.csv` (no header row)

---

## Model Pipeline

```
Raw Sonar CSV (208 × 60 features)
        │
        ▼
Load and Explore Data
(shape, label distribution, group statistics)
        │
        ▼
Separate Features (X) and Labels (Y)
        │
        ▼
Stratified Train/Test Split (90% / 10%)
        │
        ▼
Train Logistic Regression Classifier
        │
        ├──── Training Accuracy
        └──── Test Accuracy
                │
                ▼
        New Input (60 values) → Prediction → Rock or Mine
```

---

## Project Structure

```
SONAR_ROCK_VS_MINE_PREDICTOR/
│
├── SONAR_ROCK_VS_MINE_PREDICTOR/
│   ├── Copy of sonar data.csv      # Dataset (208 samples, 60 features)
│   ├── main.py                     # Full training and prediction script
│   └── readme.md                   # Inner documentation
│
└── README.md                       # Project documentation
```

---

## Installation

**Prerequisites**: Python 3.8 or higher

1. Clone the repository:

```bash
git clone https://github.com/mohankiran18/SONAR_ROCK_VS_MINE_PREDICTOR.git
cd SONAR_ROCK_VS_MINE_PREDICTOR/SONAR_ROCK_VS_MINE_PREDICTOR
```

2. Install dependencies:

```bash
pip install numpy pandas scikit-learn
```

---

## Usage

Run the prediction script:

```bash
python main.py
```

To classify a new object, provide 60 comma-separated sonar frequency values as input. The model will output either:

```
The object is a Rock.
```

or

```
The object is a Mine.
```

**Example input** (60 values):

```
0.0453, 0.0523, 0.0843, 0.0689, ...
```

---

## Model Details

**Algorithm**: Logistic Regression (Scikit-learn)

Logistic Regression was selected for its interpretability and strong baseline performance on linearly separable binary classification tasks. The dataset is split with stratification to preserve the class distribution across both training and test sets.

| Split | Size |
|---|---|
| Training | 90% (≈ 187 samples) |
| Test | 10% (≈ 21 samples) |

| Metric | Value |
|---|---|
| Training Accuracy | _Add after evaluation_ |
| Test Accuracy | _Add after evaluation_ |

> Update these values by printing `accuracy_score` results from your script and pasting them here.

---

## Key Steps in the Code

1. Load the CSV dataset using Pandas (no header)
2. Explore shape, statistical summary, and label distribution
3. Separate features (`X` = columns 0–59) and labels (`Y` = column 60)
4. Stratified train/test split (90/10)
5. Train Logistic Regression on training data
6. Evaluate accuracy on both training and test sets
7. Accept new 60-value input, reshape, and output prediction

---

## Future Improvements

- Add a Streamlit interface for interactive sonar input and prediction
- Benchmark against other classifiers (SVM, Random Forest, KNN) and compare accuracy
- Add cross-validation for more reliable performance estimates
- Visualize feature importance and correlation heatmaps
- Package as a REST API for integration with sonar hardware systems

---

## Author

**Mohan Kiran**  
B.Tech — Artificial Intelligence and Machine Learning  
GitHub: [github.com/mohankiran18](https://github.com/mohankiran18)  
Portfolio: [mohan-kiran.netlify.app](https://mohan-kiran.netlify.app/)

---

## License

This project is licensed under the [MIT License](LICENSE).

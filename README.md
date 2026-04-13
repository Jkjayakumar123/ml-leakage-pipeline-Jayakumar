# ML Leakage Detection and Pipeline Evaluation

## 📌 Project Overview

This project demonstrates how **data leakage** can occur in machine learning workflows and how to fix it using a **Pipeline and Cross-Validation**.

It also evaluates **Decision Tree depth** to understand **underfitting and overfitting** behavior.

The workflow compares:

* ❌ A flawed approach that causes **data leakage**
* ✅ A corrected approach using **Pipeline**
* 🌳 Decision Tree models with different depths

---

# 🧠 Learning Objectives

After completing this project, you will understand:

* What **data leakage** is
* Why scaling before splitting is incorrect
* How to fix leakage using **Pipeline**
* How to use **Cross-Validation**
* How **Decision Tree depth** affects model performance
* How to balance **underfitting vs overfitting**

---

# 📂 Dataset

A synthetic dataset is generated using:

```python
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=1000,
    n_features=10,
    random_state=42
)
```

Dataset Details:

* Samples: **1000**
* Features: **10**
* Type: **Binary Classification**
* Source: **Synthetic (generated programmatically)**

---

# ⚠️ Task 1 — Demonstrating Data Leakage

## ❌ Flawed Workflow

Steps performed:

1. Scale the entire dataset
2. Split into train/test
3. Train Logistic Regression
4. Evaluate accuracy

Problem:

Scaling was applied **before splitting**, causing:

* Test data influencing training
* Unrealistic performance results
* Data leakage

Example Output:

```
Train Accuracy (with leakage): 0.86
Test Accuracy  (with leakage): 0.85
```

## 🔎 Why This is Wrong

Scaling computes:

* Mean
* Standard deviation

If done before splitting:

* Test data statistics leak into training
* Model sees future information

Correct Rule:

```
Split first → Scale later
```

---

# ✅ Task 2 — Fix Using Pipeline and Cross-Validation

## Correct Workflow

Steps:

1. Split raw data first
2. Create Pipeline:

   * StandardScaler
   * LogisticRegression
3. Run 5-fold Cross-Validation
4. Compute mean accuracy

Pipeline Example:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])
```

Cross-Validation:

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    pipeline,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy"
)
```

Example Output:

```
Cross-Validation Scores:
[0.85 0.83 0.84 0.86 0.82]

Mean Accuracy: 0.84
Standard Deviation: 0.014
```

## 🎯 Benefits of Pipeline

* Prevents data leakage
* Ensures correct preprocessing
* Improves reliability
* Makes workflow reusable

---

# 🌳 Task 3 — Decision Tree Depth Experiment

Decision Tree models were trained using:

* max_depth = 1
* max_depth = 5
* max_depth = 20

## Results Table

| Depth | Train Accuracy | Test Accuracy | Interpretation |
| ----- | -------------- | ------------- | -------------- |
| 1     | Low            | Low           | Underfitting   |
| 5     | High           | High          | Best Fit       |
| 20    | Very High      | Lower         | Overfitting    |

## 📊 Observations

### Depth = 1

* Model too simple
* Cannot capture patterns
* Underfitting occurs

### Depth = 5

* Good balance
* Captures meaningful patterns
* Generalizes well

### Depth = 20

* Model too complex
* Memorizes training data
* Overfitting occurs

## ✅ Best Depth Choice

```
max_depth = 5
```

Reason:

* Good test accuracy
* Minimal overfitting
* Balanced model complexity

---

# 🛠 Installation Guide

If you encounter dependency errors, install compatible versions:

```bash
pip uninstall numpy pandas scikit-learn -y

pip install numpy==1.26.4
pip install pandas==2.2.2
pip install scikit-learn==1.4.2
```

Recommended:

Use a virtual environment:

```bash
python -m venv ml_env

ml_env\Scripts\activate

pip install numpy pandas scikit-learn
```

---

# ▶️ How to Run

Run the Python script:

```bash
python ml_leakage.py
```

Expected Output Sections:

```
Task 1: Data Leakage Example
Task 2: Pipeline Cross-Validation
Task 3: Decision Tree Comparison
Final Observations
```

---

# 📊 Key Concepts Used

* Logistic Regression
* Decision Tree Classifier
* StandardScaler
* Pipeline
* Cross-Validation
* Accuracy Metrics
* Overfitting
* Underfitting
* Data Leakage Prevention

---

# 🚀 Project Structure

```
ml-leakage-pipeline/
│
├── ml_leakage.py
├── README.md
└── requirements.txt (optional)
```

---

# 📌 Summary

This project demonstrates:

* How **data leakage happens**
* How **Pipeline prevents leakage**
* How **Cross-Validation improves reliability**
* How **Decision Tree depth affects generalization**

These practices are essential for building **trustworthy machine learning models** in real-world fintech and data science workflows.

---

# 📎 Optional Requirements File

You may include:

```
numpy==1.26.4
pandas==2.2.2
scikit-learn==1.4.2
```

Install using:

```bash
pip install -r requirements.txt
```

---

# 👨‍💻 Author

Junior Data Science Workflow Demonstration
Machine Learning Best Practices Project

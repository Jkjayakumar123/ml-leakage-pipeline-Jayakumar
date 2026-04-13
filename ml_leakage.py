# ============================================================
# Model Evaluation and Data Leakage Demonstration
# Tasks Covered:
# Task 1 — Demonstrate Data Leakage
# Task 2 — Fix Workflow using Pipeline + Cross-Validation
# Task 3 — Decision Tree Depth Experiment
# ============================================================

# Import Libraries
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# ============================================================
# Generate Dataset
# ============================================================

X, y = make_classification(
    n_samples=1000,
    n_features=10,
    random_state=42
)

# ============================================================
# Task 1 — Flawed Workflow (Data Leakage)
# ============================================================

print("\n========== Task 1: Data Leakage Example ==========")

# ❌ WRONG: Scaling entire dataset before splitting
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split after scaling
X_train_leak, X_test_leak, y_train_leak, y_test_leak = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# Train Logistic Regression
model_leak = LogisticRegression(max_iter=1000)
model_leak.fit(X_train_leak, y_train_leak)

# Predictions
train_pred_leak = model_leak.predict(X_train_leak)
test_pred_leak = model_leak.predict(X_test_leak)

# Accuracy
train_acc_leak = accuracy_score(y_train_leak, train_pred_leak)
test_acc_leak = accuracy_score(y_test_leak, test_pred_leak)

print("Train Accuracy (with leakage):", train_acc_leak)
print("Test Accuracy  (with leakage):", test_acc_leak)

print("\nProblem Identified:")
print("Scaling was done before splitting the dataset.")
print("This caused data leakage because test data influenced scaling.")

# ============================================================
# Task 2 — Correct Workflow using Pipeline + Cross Validation
# ============================================================

print("\n========== Task 2: Correct Pipeline Workflow ==========")

# Correct split (raw data)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

# Cross Validation
cv_scores = cross_val_score(
    pipeline,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy"
)

print("Cross-Validation Scores:", cv_scores)
print("Mean Accuracy:", cv_scores.mean())
print("Standard Deviation:", cv_scores.std())

# ============================================================
# Task 3 — Decision Tree Depth Experiment
# ============================================================

print("\n========== Task 3: Decision Tree Depth Comparison ==========")

depths = [1, 5, 20]

print("\nDepth | Train Accuracy | Test Accuracy")
print("----------------------------------------")

for depth in depths:

    tree = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    tree.fit(X_train, y_train)

    train_acc = tree.score(X_train, y_train)
    test_acc = tree.score(X_test, y_test)

    print(depth, "   |", round(train_acc, 4), "        |", round(test_acc, 4))

# ============================================================
# Final Explanation Output
# ============================================================

print("\n========== Final Observations ==========")

print("""
Depth = 1  → Underfitting (model too simple)

Depth = 5  → Best balance between training and testing accuracy

Depth = 20 → Overfitting (model memorizes training data)

Recommended Depth: 5
""")
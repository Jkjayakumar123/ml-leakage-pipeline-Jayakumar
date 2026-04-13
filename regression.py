# ============================================================
# House Price Prediction using Multiple Linear Regression
# Tasks Covered:
# Task 1 — Create Dataset & Train Model
# Task 2 — Evaluate Model (MAE, RMSE, R2)
# Task 3 — Residual Analysis with Histogram
# ============================================================

# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# ============================================================
# Task 1 — Create Synthetic Dataset
# ============================================================

np.random.seed(42)

# Generate synthetic data (100 records)
n = 100

area_sqft = np.random.randint(500, 3000, n)
num_bedrooms = np.random.randint(1, 6, n)
age_years = np.random.randint(0, 30, n)

# Create realistic price formula (with noise)
price_lakhs = (
    area_sqft * 0.05 +
    num_bedrooms * 10 -
    age_years * 0.3 +
    np.random.normal(0, 10, n)
)

# Create DataFrame
df = pd.DataFrame({
    "area_sqft": area_sqft,
    "num_bedrooms": num_bedrooms,
    "age_years": age_years,
    "price_lakhs": price_lakhs
})

print("\nFirst 5 rows of dataset:\n")
print(df.head())

# Features and Target
X = df[["area_sqft", "num_bedrooms", "age_years"]]
y = df["price_lakhs"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Print Intercept and Coefficients
print("\nModel Intercept:")
print(model.intercept_)

print("\nFeature Coefficients:")

for feature, coef in zip(X.columns, model.coef_):
    print(feature, ":", coef)

# Predictions
y_pred = model.predict(X_test)

# Show Actual vs Predicted
results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print("\nFirst 5 Actual vs Predicted Values:\n")
print(results.head())

# ============================================================
# Task 2 — Model Evaluation
# ============================================================

# Calculate Metrics
mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Metrics:\n")

print("MAE  :", mae)
print("RMSE :", rmse)
print("R²   :", r2)

# ------------------------------------------------------------
# Explanation of Metrics:
#
# MAE (Mean Absolute Error):
# Shows the average absolute difference between actual and predicted prices.
# Lower MAE means predictions are closer to actual values.
#
# RMSE (Root Mean Squared Error):
# Penalizes larger errors more than MAE.
# Lower RMSE indicates fewer large prediction mistakes.
#
# R² Score:
# Indicates how much variance in house price is explained by the model.
# Values closer to 1 indicate better model performance.
# ------------------------------------------------------------

# ============================================================
# Task 3 — Residual Analysis
# ============================================================

# Calculate Residuals
residuals = y_test - y_pred

# Plot Histogram
plt.figure()

plt.hist(residuals, bins=15)

plt.title("Histogram of Residuals")

plt.xlabel("Residual Value")

plt.ylabel("Frequency")

plt.show()

# ------------------------------------------------------------
# Residual Explanation:
#
# A residual is the difference between the actual price
# and the predicted price:
#
# Residual = Actual − Predicted
#
# If the histogram looks roughly symmetrical and centered
# around zero, it suggests the model errors are balanced
# and predictions are reasonably accurate.
#
# If the histogram is skewed or widely spread,
# it indicates possible bias or poor model fit.
# ------------------------------------------------------------
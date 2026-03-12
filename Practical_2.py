# ==========================================
# PRACTICAL 2 : LINEAR REGRESSION
# AIM:
# To implement Linear Regression for predicting continuous
# target variables and evaluate the model performance using
# appropriate metrics along with visualization.
# ==========================================

# ------------------------------------------
# Step 1 : Import Required Libraries
# ------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------------------------------
# Step 2 : Load Dataset
# ------------------------------------------

# Replace this with your dataset path
DATA_PATH = "dataset.csv"

data = pd.read_csv(DATA_PATH)

print("\nDataset Loaded Successfully")

# ------------------------------------------
# Step 3 : Display Dataset
# ------------------------------------------

print("\nFirst 5 rows of dataset")
print(data.head())

# ------------------------------------------
# Step 4 : Dataset Information
# ------------------------------------------

print("\nDataset Info")
print(data.info())

# ------------------------------------------
# Step 5 : Statistical Summary
# ------------------------------------------

print("\nStatistical Summary")
print(data.describe())

# ------------------------------------------
# Step 6 : Check Missing Values
# ------------------------------------------

print("\nMissing Values")
print(data.isnull().sum())

# ------------------------------------------
# Step 7 : Data Visualization (EDA)
# ------------------------------------------

# Histogram Distribution
data.hist(figsize=(10,8))
plt.suptitle("Feature Distribution")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# Pair Plot
sns.pairplot(data)
plt.show()

# ------------------------------------------
# Step 8 : Feature and Target Selection
# ------------------------------------------

TARGET_COLUMN = data.columns[-1]

X = data.drop(columns=[TARGET_COLUMN])
y = data[TARGET_COLUMN]

print("\nFeature Columns:", X.columns)
print("Target Column:", TARGET_COLUMN)

# ------------------------------------------
# Step 9 : Train-Test Split
# ------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state = 42
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# ------------------------------------------
# Step 10 : Train Linear Regression Model
# ------------------------------------------

model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel Training Completed")

# ------------------------------------------
# Step 11 : Model Coefficients
# ------------------------------------------

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print("\nModel Coefficients")
print(coefficients)

print("\nIntercept:", model.intercept_)

# ------------------------------------------
# Step 12 : Predictions
# ------------------------------------------

predictions = model.predict(X_test)

# ------------------------------------------
# Step 13 : Performance Evaluation
# ------------------------------------------

MAE = mean_absolute_error(y_test, predictions)
MSE = mean_squared_error(y_test, predictions)
RMSE = np.sqrt(MSE)
R2 = r2_score(y_test, predictions)

print("\nModel Performance Metrics")

print("Mean Absolute Error (MAE):", MAE)
print("Mean Squared Error (MSE):", MSE)
print("Root Mean Squared Error (RMSE):", RMSE)
print("R2 Score:", R2)

# ------------------------------------------
# Step 14 : Visualization
# ------------------------------------------

# Actual vs Predicted Values

plt.figure(figsize=(6,5))
plt.scatter(y_test, predictions)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")
plt.show()

# ------------------------------------------

# Residual Plot

residuals = y_test - predictions

plt.figure(figsize=(6,5))
plt.scatter(predictions, residuals)
plt.axhline(y=0, color='red')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

# ------------------------------------------

# Residual Distribution

plt.figure(figsize=(6,5))
sns.histplot(residuals, kde=True)
plt.title("Residual Distribution")
plt.show()

# ------------------------------------------

# Regression Fit Visualization (for first feature)

if X.shape[1] == 1:
    
    plt.scatter(X_test, y_test)
    
    plt.plot(X_test, predictions, color="red")
    
    plt.xlabel(X.columns[0])
    
    plt.ylabel(TARGET_COLUMN)
    
    plt.title("Regression Line")
    
    plt.show()

# ------------------------------------------
# Step 15 : Result
# ------------------------------------------

print("\nResult:")
print("Linear Regression model implemented successfully.")
print("Continuous target variable predicted and evaluated using MAE, MSE, RMSE, and R2 metrics.")
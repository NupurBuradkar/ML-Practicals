# ==============================
# PRACTICAL 1
# AIM:
# To understand Python libraries used for Machine Learning (NumPy, Pandas,
# Matplotlib, Seaborn) by performing basic data analysis and visualization.
# ==============================

# 1) Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 2) Create a simple dataset (since the aim is to learn libraries)
data = {
    "StudyHours": [1,2,3,4,5,6,7,8],
    "Marks":      [30,35,40,50,55,65,70,80],
    "SleepHours": [7,6,7,6,8,7,6,8]
}
df = pd.DataFrame(data)

# 3) Basic Inspection
print("First 5 rows:\n", df.head())
print("\nDataset Info:")
print(df.info())
print("\nStatistical Summary:")
print(df.describe())

# 4) NumPy demonstration
arr = np.array(df["StudyHours"])
print("\nNumPy array of StudyHours:", arr)
print("Mean StudyHours using NumPy:", np.mean(arr))

# 5) Visualization — Scatter Plot
plt.scatter(df["StudyHours"], df["Marks"])
plt.title("Study Hours vs Marks")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.show()

# 6) Line Plot
plt.plot(df["StudyHours"], df["Marks"], marker='o')
plt.title("Line Plot: Study Hours vs Marks")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.show()

# 7) Histogram
df.hist(figsize=(6,4))
plt.suptitle("Feature Distributions")
plt.show()

# 8) Correlation Heatmap
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

print("\nResult: Basic data analysis and visualization completed successfully.")
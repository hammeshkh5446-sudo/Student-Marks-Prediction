# ================================
# 📌 Student Marks Prediction Project
# ================================

# ---------- 1. Import Libraries ----------
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# ---------- 2. Load Dataset ----------
df = pd.read_csv('students_real.csv')

# Preview data
print("First 5 Rows:\n", df.head())

# Dataset info (columns, types, missing values)
print("\nDataset Info:\n")
df.info()

# Statistical summary
print("\nStatistical Summary:\n", df.describe())


# ---------- 3. Feature Selection ----------
# Selecting important features based on analysis
X = df[["Hours", "Attendance"]]   # Independent variables
y = df["Marks"]                  # Target variable


# ---------- 4. Split Data ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ---------- 5. Train Model ----------
model = LinearRegression()
model.fit(X_train, y_train)


# ---------- 6. Data Visualization ----------

# Heatmap to check feature correlation
plt.figure()
sns.heatmap(df.corr(), annot=True)
plt.title("Feature Correlation Heatmap")
plt.show()

# Scatter plot: Hours vs Marks
plt.figure()
sns.scatterplot(x=df["Hours"], y=df["Marks"])
plt.title("Hours vs Marks")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.show()


# ---------- 7. Model Evaluation ----------
y_pred = model.predict(X_test)
score = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("R2 Score:", score)


# ---------- 8. Single Prediction ----------
new_data = [[6, 80]]  # Hours, Attendance
prediction = model.predict(new_data)

print("\nSingle Prediction:")
print("Input:", new_data)
print("Predicted Marks:", prediction)


# ---------- 9. Multiple Test Cases ----------
test_cases = [
    [4, 70],
    [7, 85],
    [10, 92]
]

print("\nMultiple Predictions:")

for case in test_cases:
    pred = model.predict([case])
    print(f"Input: {case} → Predicted Marks: {pred[0]}")

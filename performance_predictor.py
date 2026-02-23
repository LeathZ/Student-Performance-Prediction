# ==========================================================
# Student Performance Predictor
# Dataset: UCI Student Performance (student-mat.csv)
# Algorithm: Decision Tree Classifier
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# ----------------------------------------------------------
# 1. LOAD DATASET
# ----------------------------------------------------------

print("Loading dataset...")

df = pd.read_csv(R"C:\Users\LeathLOQ\Downloads\student\student-mat.csv", sep=";")

print("Dataset loaded successfully.")
print("Total records:", len(df))
print("Total features:", len(df.columns))

# ----------------------------------------------------------
# 2. INITIAL DATA INSPECTION
# ----------------------------------------------------------

print("\n--- Initial Data (Head) ---")
print(df.head())

print("\n--- Initial Data (Info) ---")
print(df.info())

print("\n--- Initial Data (Missing Values) ---")
print(df.isnull().sum())

# ----------------------------------------------------------
# 3. DATA CLEANING AND PREPROCESSING
# ----------------------------------------------------------

print("\n--- Data Cleaning ---")

# Create binary target variable: Pass (1) if G3 >= 10, Fail (0) otherwise
df["pass"] = (df["G3"] >= 10).astype(int)
print("Target variable created: Pass =", df["pass"].sum(), "| Fail =", (df["pass"] == 0).sum())

# Drop G1, G2, G3 to avoid data leakage (G3 is what we're predicting)
df.drop(columns=["G1", "G2", "G3"], inplace=True)

# Separate categorical and numerical columns
categorical_cols = df.select_dtypes(include="object").columns.tolist()
numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
numerical_cols.remove("pass")  # remove target from features

print("\nCategorical columns:", categorical_cols)
print("Numerical columns:", numerical_cols)

# Label encode categorical columns
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

print("\n--- Data After Encoding (Head) ---")
print(df.head())

# Normalize numerical columns
scaler = MinMaxScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

print("\n--- Data After Normalization (Head, numerical columns) ---")
print(df[numerical_cols].head())

print("\n--- Data After Normalization (Descriptive Stats) ---")
print(df[numerical_cols].describe())

# ----------------------------------------------------------
# 4. SPLIT INTO TRAIN, VALIDATION, TEST SETS
# ----------------------------------------------------------

X = df.drop(columns=["pass"])
y = df["pass"]

# First split: 80% train+val, 20% test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Second split: 75% train, 25% validation (of the 80%)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
)

print("\n--- Dataset Split ---")
print(f"Training set:   {len(X_train)} records ({len(X_train)/len(X)*100:.1f}%)")
print(f"Validation set: {len(X_val)} records ({len(X_val)/len(X)*100:.1f}%)")
print(f"Test set:       {len(X_test)} records ({len(X_test)/len(X)*100:.1f}%)")

# ----------------------------------------------------------
# 5. HYPERPARAMETER TUNING (max_depth)
# ----------------------------------------------------------

print("\n--- Hyperparameter Tuning (max_depth) ---")

depth_range = range(1, 15)
train_scores = []
val_scores = []

for depth in depth_range:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)
    train_scores.append(accuracy_score(y_train, dt.predict(X_train)))
    val_scores.append(accuracy_score(y_val, dt.predict(X_val)))

# Find best depth
best_depth = depth_range[val_scores.index(max(val_scores))]
print(f"Best max_depth: {best_depth} (Validation Accuracy: {max(val_scores):.4f})")

# Plot max_depth vs accuracy
plt.figure(figsize=(8, 5))
plt.plot(depth_range, train_scores, label="Train Accuracy", marker="o")
plt.plot(depth_range, val_scores, label="Validation Accuracy", marker="s")
plt.axvline(x=best_depth, color="red", linestyle="--", label=f"Best depth = {best_depth}")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.title("Decision Tree: max_depth vs Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("hyperparameter_tuning.png", dpi=150)
plt.show()
print("Saved: hyperparameter_tuning.png")

# ----------------------------------------------------------
# 6. TRAIN FINAL MODEL
# ----------------------------------------------------------

print("\n--- Training Final Model ---")

best_dt = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
best_dt.fit(X_train, y_train)

print("Model trained successfully.")

# ----------------------------------------------------------
# 7. CROSS-VALIDATION
# ----------------------------------------------------------

cv_scores = cross_val_score(best_dt, X_train_val, y_train_val, cv=5, scoring="accuracy")
print(f"\n--- 5-Fold Cross-Validation ---")
print(f"Scores: {cv_scores.round(4)}")
print(f"Mean: {cv_scores.mean():.4f} | Std: {cv_scores.std():.4f}")

# ----------------------------------------------------------
# 8. EVALUATION ON TEST SET
# ----------------------------------------------------------

y_pred = best_dt.predict(X_test)

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)

print("\n--- Final Model Performance (Test Set) ---")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Fail", "Pass"]))

# ----------------------------------------------------------
# 9. CONFUSION MATRIX
# ----------------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Fail", "Pass"],
            yticklabels=["Fail", "Pass"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("Saved: confusion_matrix.png")

# ----------------------------------------------------------
# 10. FEATURE IMPORTANCE
# ----------------------------------------------------------

importances = best_dt.feature_importances_
feature_names = X.columns
feat_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
feat_df = feat_df.sort_values("Importance", ascending=False).head(10)

plt.figure(figsize=(8, 5))
sns.barplot(x="Importance", y="Feature", data=feat_df, palette="viridis")
plt.title("Top 10 Feature Importances")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()
print("Saved: feature_importance.png")

# ----------------------------------------------------------
# 11. DECISION TREE VISUALIZATION
# ----------------------------------------------------------

plt.figure(figsize=(20, 8))
plot_tree(best_dt, feature_names=X.columns,
          class_names=["Fail", "Pass"],
          filled=True, rounded=True, max_depth=3)
plt.title("Decision Tree (max_depth shown: 3)")
plt.tight_layout()
plt.savefig("decision_tree.png", dpi=150)
plt.show()
print("Saved: decision_tree.png")

print("\n✅ All done!")
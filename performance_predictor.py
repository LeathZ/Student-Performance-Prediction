# ==========================================================
# Student Performance Predictor
# Dataset: UCI Student Performance (student-mat.csv)
# Algorithm: Decision Tree Classifier
# Features: failures, studytime, absences, goout
# ==========================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# Save images to the same folder as this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
def save(filename):
    path = os.path.join(SCRIPT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")

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

selected_raw = ["failures", "studytime", "absences", "goout", "G3"]

print("\n--- Initial Data (Head) ---")
print(df[selected_raw].head())

print("\n--- Initial Data (Info) ---")
print(df[selected_raw].info())

print("\n--- Initial Data (Missing Values) ---")
print(df[selected_raw].isnull().sum())

# ----------------------------------------------------------
# 3. DATA CLEANING AND PREPROCESSING
# ----------------------------------------------------------

print("\n--- Data Cleaning ---")

# Create binary target variable
df["pass"] = (df["G3"] >= 10).astype(int)
print("Target variable created: Pass =", df["pass"].sum(), "| Fail =", (df["pass"] == 0).sum())

# Select only the 4 features + target
features = ["failures", "studytime", "absences", "goout"]
df = df[features + ["pass"]].copy()

print("\nFeatures selected:", features)

# Normalize the 4 numerical features
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

print("\n--- Data After Normalization (Head) ---")
print(df[features].head())

print("\n--- Data After Normalization (Descriptive Stats) ---")
print(df[features].describe())

# ----------------------------------------------------------
# 4. SPLIT INTO TRAIN, VALIDATION, TEST SETS
# ----------------------------------------------------------

X = df[features]
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

best_depth = depth_range[val_scores.index(max(val_scores))]
print(f"Best max_depth: {best_depth} (Validation Accuracy: {max(val_scores):.4f})")

# Plot
plt.figure(figsize=(8, 5))
plt.plot(depth_range, train_scores, label="Train Accuracy", marker="o")
plt.plot(depth_range, val_scores, label="Validation Accuracy", marker="s")
plt.axvline(x=best_depth, color="red", linestyle="--", label=f"Best depth = {best_depth}")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.title("Decision Tree: max_depth vs Accuracy")
plt.legend()
plt.tight_layout()
save("hyperparameter_tuning.png")
plt.show()

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
save("confusion_matrix.png")
plt.show()

# ----------------------------------------------------------
# 10. FEATURE IMPORTANCE
# ----------------------------------------------------------

importances = best_dt.feature_importances_
feat_df = pd.DataFrame({"Feature": features, "Importance": importances})
feat_df = feat_df.sort_values("Importance", ascending=False)

plt.figure(figsize=(7, 4))
sns.barplot(x="Importance", y="Feature", data=feat_df,
            hue="Feature", palette="viridis", legend=False)
plt.title("Feature Importances")
plt.tight_layout()
save("feature_importance.png")
plt.show()

# ----------------------------------------------------------
# 11. DECISION TREE VISUALIZATION
# ----------------------------------------------------------

plt.figure(figsize=(16, 8))
plot_tree(best_dt, feature_names=features,
          class_names=["Fail", "Pass"],
          filled=True, rounded=True)
plt.title(f"Decision Tree (max_depth = {best_depth})")
plt.tight_layout()
save("decision_tree.png")
plt.show()

print("\n✅ All done!")
from preprocess import prepare_data

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib
from evaluate import evaluate_model

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# PREPARE DATA
# ============================================================

X_train_processed, X_test_processed, y_train, y_test, preprocessor = prepare_data()


# ============================================================
# ============================================================
# 1. LOGISTIC REGRESSION
# ============================================================
# ============================================================

# ------------------------------------------------------------
# Create Model
# ------------------------------------------------------------

logistic_model = LogisticRegression(
    max_iter=300,
    solver="liblinear",
    random_state=42
)


# ------------------------------------------------------------
# Train Model
# ------------------------------------------------------------

logistic_model.fit(
    X_train_processed,
    y_train
)


# ------------------------------------------------------------
# Make Predictions
# ------------------------------------------------------------

logistic_pred = logistic_model.predict(
    X_test_processed
)


# ------------------------------------------------------------
# Evaluate Model
# ------------------------------------------------------------

logistic_results = evaluate_model(
    y_test,
    logistic_pred,
    "Logistic Regression"
)


# ============================================================
# ============================================================
# 2. DECISION TREE
# ============================================================
# ============================================================

# ------------------------------------------------------------
# Create Model
# ------------------------------------------------------------

tree_model = DecisionTreeClassifier(
    random_state=42
)


# ------------------------------------------------------------
# Train Model
# ------------------------------------------------------------

tree_model.fit(
    X_train_processed,
    y_train
)

# ------------------------------------------------------------
# Save Trained Model
# ------------------------------------------------------------

joblib.dump(
    tree_model,
    "models/decision_tree.pkl"
)

joblib.dump(
    preprocessor,
    "models/preprocessor.pkl"
)
# ------------------------------------------------------------
# Make Predictions
# ------------------------------------------------------------

tree_pred = tree_model.predict(
    X_test_processed
)


# ------------------------------------------------------------
# Evaluate Model
# ------------------------------------------------------------

tree_results = evaluate_model(
    y_test,
    tree_pred,
    "Decision Tree"
)


# ============================================================
# ============================================================
# 3. RANDOM FOREST
# ============================================================
# ============================================================

# ------------------------------------------------------------
# Create Model
# ------------------------------------------------------------

forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# ------------------------------------------------------------
# Train Model
# ------------------------------------------------------------

forest_model.fit(
    X_train_processed,
    y_train
)


# ------------------------------------------------------------
# Make Predictions
# ------------------------------------------------------------

forest_pred = forest_model.predict(
    X_test_processed
)


# ------------------------------------------------------------
# Evaluate Model
# ------------------------------------------------------------

forest_results = evaluate_model(
    y_test,
    forest_pred,
    "Random Forest"
)


# ============================================================
# COMPARE MODELS
# ============================================================

results = pd.DataFrame([
    logistic_results,
    tree_results,
    forest_results
])

print("\n")
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(results.to_string(index=False))


# ============================================================
# CONFUSION MATRIX COMPARISON
# ============================================================

model_results = [
    logistic_results,
    tree_results,
    forest_results
]

fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 5)
)

for ax, result in zip(axes, model_results):

    sns.heatmap(
        result["Confusion Matrix"],
        annot=True,
        fmt="d",
        xticklabels=["Stay", "Churn"],
        yticklabels=["Stay", "Churn"],
        ax=ax
    )

    ax.set_title(result["Model"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")


plt.suptitle(
    "Confusion Matrix Comparison",
    fontsize=16
)

plt.tight_layout()
plt.show()
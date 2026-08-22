from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    recall_score,
    precision_score,
    f1_score
)

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# Evaluate a Classification Model
# ============================================================

def evaluate_model(y_test, y_pred, model_name):

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n{'=' * 50}")
    print(f"{model_name} Evaluation")
    print(f"{'=' * 50}")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1-Score : {f1:.4f}")

    cm = confusion_matrix(y_test, y_pred)

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Recall": recall,
        "Precision": precision,
        "F1-Score": f1,
        "Confusion Matrix": cm
    }
import joblib
import pandas as pd
from sklearn.tree import export_text

# ============================================================
# 1. Load Saved Preprocessor
# ============================================================

preprocessor = joblib.load(
    "models/preprocessor.pkl"
)


# ============================================================
# 2. Load Saved Decision Tree
# ============================================================

model = joblib.load(
    "models/decision_tree.pkl"
)


# ============================================================
# 3. Create New Customer
# ============================================================

new_customer = pd.DataFrame([{
    "Age": 35,
    "Gender": "Female",
    "Tenure": 20,
    "Usage Frequency": 15,
    "Support Calls": 8,
    "Payment Delay": 23,
    "Subscription Type": "Standard",
    "Contract Length": "Monthly",
    "Total Spend": 500,
    "Last Interaction": 10
}])


# ============================================================
# 4. Preprocess New Customer
# ============================================================

new_customer_processed = preprocessor.transform(
    new_customer
)


# ============================================================
# 5. Make Prediction
# ============================================================

prediction = model.predict(
    new_customer_processed
)

# ============================================================
# Inspect Decision Tree
# ============================================================
tree_rules = export_text(
    model,
    feature_names=preprocessor.get_feature_names_out()
)

print("\nDecision Tree Rules:")
print(tree_rules)

# ============================================================
# 6. Display Result
# ============================================================

if prediction[0] == 1:
    print("Prediction: Churn")
else:
    print("Prediction: Stay")




# Raw Dataset
#     ↓
# EDA
#     ↓
# Preprocessing
#     ↓
# Train / Test Split
#     ↓
# Train 3 Models
#     ↓
# Predict on Test Set
#     ↓
# Evaluate
#     ↓
# Compare Models
#     ↓
# Select Decision Tree
#     ↓
# Save Model + Preprocessor
#     ↓
# New Customer
#     ↓
# Load Saved Model + Preprocessor
#     ↓
# Transform New Customer
#     ↓
# Predict
#     ↓
# CHURN    
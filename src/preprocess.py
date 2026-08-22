# ============================================================
# FEATURES
# X → Customer Features
#       │
#       ▼
# Classification Model
#       │
#       ▼
# Churn
# y → Target
# ============================================================


import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


# ============================================================
# Prepare Data
# ============================================================

def prepare_data():

    # ========================================================
    # 1. Load Data
    # ========================================================

    df = pd.read_csv(
        "data/customer_churn_dataset-training-master.csv/"
        "customer_churn_dataset-training-master.csv"
    )


    # ========================================================
    # 2. Remove Missing Rows
    # ========================================================

    df = df.dropna()


    # ========================================================
    # 3. Remove Non-Predictive Columns
    # ========================================================

    df = df.drop(
        columns=["CustomerID"],
        errors="ignore"
    )


    # ========================================================
    # 4. Separate Features and Target
    # ========================================================

    X = df.drop(columns=["Churn"])
    y = df["Churn"]


    # ========================================================
    # 5. Identify Feature Types
    # ========================================================

    categorical_features = [
        "Gender",
        "Subscription Type",
        "Contract Length"
    ]

    numerical_features = [
        "Age",
        "Tenure",
        "Usage Frequency",
        "Support Calls",
        "Payment Delay",
        "Total Spend",
        "Last Interaction"
    ]


    # ========================================================
    # 6. Encode Categorical Features
    # ========================================================

    encoder = OneHotEncoder(
        handle_unknown="ignore"
    )


    # ========================================================
    # 7. Create Preprocessing Transformer
    # ========================================================

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                encoder,
                categorical_features
            )
        ],
        remainder="passthrough"
    )


    # ========================================================
    # 8. Split Data into Training and Testing Sets
    # ========================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


    # ========================================================
    # 9. Fit Preprocessor on Training Data
    # ========================================================

    X_train_processed = preprocessor.fit_transform(
        X_train
    )


    # ========================================================
    # 10. Transform Test Data
    # ========================================================

    X_test_processed = preprocessor.transform(
        X_test
    )


    # fit:
    # Learn the rules from the data.
    #
    # transform:
    # Apply the rules that were already learned.
    #
    # fit_transform:
    # Learn the rules + apply them to the same data.


    # ========================================================
    # 11. Check Processed Data
    # ========================================================

    print("\nOriginal training shape:")
    print(X_train.shape)

    print("\nProcessed training shape:")
    print(X_train_processed.shape)

    print("\nOriginal test shape:")
    print(X_test.shape)

    print("\nProcessed test shape:")
    print(X_test_processed.shape)


    # ========================================================
    # 12. Return Prepared Data
    # ========================================================

    return (
    X_train_processed,
    X_test_processed,
    y_train,
    y_test,
    preprocessor
)
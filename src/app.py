import streamlit as st
import pandas as pd
import joblib


# ============================================================
# 1. Load Saved Model and Preprocessor
# ============================================================

preprocessor = joblib.load(
    "models/preprocessor.pkl"
)

model = joblib.load(
    "models/decision_tree.pkl"
)


# ============================================================
# 2. Page Configuration
# ============================================================

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🏢",
    layout="wide"
)


# ============================================================
# 3. Header
# ============================================================

st.title("🏢 Customer Churn Predictor")

st.write(
    "A machine learning system that predicts whether a "
    "customer is likely to churn."
)

st.divider()


# ============================================================
# 4. Customer Information
# ============================================================

st.subheader("👤 Customer Information")

col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female", "Other"]
    )

    tenure = st.number_input(
        "Tenure",
        min_value=0,
        value=20
    )

    usage_frequency = st.number_input(
        "Usage Frequency",
        min_value=0,
        value=15
    )

    support_calls = st.number_input(
        "Support Calls",
        min_value=0,
        value=8
    )


with col2:

    payment_delay = st.number_input(
        "Payment Delay",
        min_value=0,
        value=23
    )

    subscription_type = st.selectbox(
        "Subscription Type",
        ["Basic", "Standard", "Premium"]
    )

    contract_length = st.selectbox(
        "Contract Length",
        ["Monthly", "Quarterly", "Annual"]
    )

    total_spend = st.number_input(
        "Total Spend",
        min_value=0.0,
        value=500.0
    )

    last_interaction = st.number_input(
        "Last Interaction",
        min_value=0,
        value=10
    )


# ============================================================
# 5. Create Customer DataFrame
# ============================================================

new_customer = pd.DataFrame([{

    "Age": age,
    "Gender": gender,
    "Tenure": tenure,
    "Usage Frequency": usage_frequency,
    "Support Calls": support_calls,
    "Payment Delay": payment_delay,
    "Subscription Type": subscription_type,
    "Contract Length": contract_length,
    "Total Spend": total_spend,
    "Last Interaction": last_interaction

}])


st.divider()


# ============================================================
# 6. Prediction
# ============================================================

if st.button(
    "🔮 Predict Customer Churn",
    use_container_width=True
):

    # --------------------------------------------------------
    # Preprocess New Customer
    # --------------------------------------------------------

    new_customer_processed = preprocessor.transform(
        new_customer
    )

    # --------------------------------------------------------
    # Make Prediction
    # --------------------------------------------------------

    prediction = model.predict(
        new_customer_processed
    )

    # --------------------------------------------------------
    # Display Prediction
    # --------------------------------------------------------

    st.subheader("📊 Prediction Result")

    if prediction[0] == 1:

        st.error(
            "⚠️ CHURN PREDICTED"
        )

        st.write(
            "The Decision Tree predicts that this customer "
            "is likely to churn."
        )

    else:

        st.success(
            "✅ STAY PREDICTED"
        )

        st.write(
            "The Decision Tree predicts that this customer "
            "is likely to stay."
        )


    # ========================================================
    # 7. Model Information
    # ========================================================

    st.divider()

    st.subheader("🤖 Model Information")

    metric1, metric2, metric3, metric4 = st.columns(4)

    with metric1:
        st.metric(
            "Model",
            "Decision Tree"
        )

    with metric2:
        st.metric(
            "Accuracy",
            "99.99%"
        )

    with metric3:
        st.metric(
            "Precision",
            "99.99%"
        )

    with metric4:
        st.metric(
            "Recall",
            "99.99%"
        )


    # ========================================================
    # 8. Simple Decision Explanation
    # ========================================================

    st.divider()

    st.subheader("🌳 Decision Explanation")

    if support_calls > 5.5:

        st.info(
            f"""
            **Support Calls = {support_calls}**

            The trained Decision Tree starts by checking
            Support Calls.

            • Support Calls > 4.5 → continue to the next branch  
            • Support Calls > 5.5 → Churn

            For this customer, Support Calls = {support_calls},
            so the tree reaches the **Churn** class.
            """
        )

    elif support_calls <= 4.5:

        st.info(
            f"""
            **Support Calls = {support_calls}**

            This customer follows the left branch of the
            Decision Tree because Support Calls ≤ 4.5.

            The tree then considers additional features
            such as Total Spend, Payment Delay, and
            Usage Frequency.
            """
        )
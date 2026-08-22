import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Churn meaning
# ============================================================
# Churn = 0 → Customer stays with the service
# Churn = 1 → Customer stops using / leaves the service


# ============================================================
# 1. Load Data
# ============================================================

df = pd.read_csv(
    "data/customer_churn_dataset-training-master.csv/"
    "customer_churn_dataset-training-master.csv"
)


# ============================================================
# 2. Initial Data Understanding
# ============================================================

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

summary = pd.DataFrame({
    "Column": df.columns,
    "Data Type": df.dtypes.astype(str).values,
    "Missing Values": df.isnull().sum().values,
    "Unique Values": df.nunique().values
})

print("\nDataset Summary:")
print(summary.to_string(index=False))

print("\nChurn Distribution:")
print(df["Churn"].value_counts())

print("\nChurn Proportion:")
print(df["Churn"].value_counts(normalize=True))


# ============================================================
# 3. Data Cleaning
# ============================================================

df = df.dropna()

print("\nAfter removing missing rows:")
print("Shape:", df.shape)

print("\nMissing values after cleaning:")
print(df.isnull().sum())


# ============================================================
# 4. Exploratory Data Analysis (EDA)
# Every visualization should answer a question about churn.
# ============================================================


# ============================================================
# EDA 1: Churn Distribution
# Question: How is churn distributed in the dataset?
# ============================================================

churn_counts = df["Churn"].value_counts()

print("\nChurn Distribution:")
print(churn_counts)


# ============================================================
# EDA 2: Churn by Contract Length
# Question: Is churn associated with contract length?
# ============================================================

contract_churn = pd.crosstab(
    df["Contract Length"],
    df["Churn"]
)

print("\nChurn by Contract Length:")
print(contract_churn)

contract_churn_rate = pd.crosstab(
    df["Contract Length"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Rate by Contract Length (%):")
print(contract_churn_rate)


# ============================================================
# EDA 3: Churn by Support Calls
# Question: Is churn associated with the number of support calls?
# ============================================================

support_calls_churn = pd.crosstab(
    df["Support Calls"],
    df["Churn"]
)

print("\nChurn by Support Calls:")
print(support_calls_churn)

support_calls_churn_rate = pd.crosstab(
    df["Support Calls"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Rate by Support Calls (%):")
print(support_calls_churn_rate)


# ============================================================
# EDA 4: Churn by Payment Delay
# Question: Does churn rate change as payment delay increases?
# ============================================================

payment_delay_churn_rate = pd.crosstab(
    df["Payment Delay"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Rate by Payment Delay (%):")
print(payment_delay_churn_rate)


# ============================================================
# EDA 5: Churn by Subscription Type
# Question: Is churn associated with subscription type?
# ============================================================

subscription_churn = pd.crosstab(
    df["Subscription Type"],
    df["Churn"]
)

print("\nChurn by Subscription Type:")
print(subscription_churn)

subscription_churn_rate = pd.crosstab(
    df["Subscription Type"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Rate by Subscription Type (%):")
print(subscription_churn_rate)


# ============================================================
# EDA 6: Churn by Usage Frequency
# Question: Is churn associated with usage frequency?
# ============================================================

usage_churn = pd.crosstab(
    df["Usage Frequency"],
    df["Churn"]
)

print("\nChurn by Usage Frequency:")
print(usage_churn)

usage_churn_rate = pd.crosstab(
    df["Usage Frequency"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Rate by Usage Frequency (%):")
print(usage_churn_rate)


# ============================================================
# EDA 7: Churn by Gender
# Question: Is churn associated with gender?
# ============================================================

gender_churn_rate = pd.crosstab(
    df["Gender"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Rate by Gender (%):")
print(gender_churn_rate)


# ============================================================
# EDA 8: Churn by Tenure
# Question: Does churn change depending on how long the
# customer has been with the company?
# ============================================================

tenure_churn_rate = pd.crosstab(
    df["Tenure"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Rate by Tenure (%):")
print(tenure_churn_rate)


# ============================================================
# EDA 9: Churn by Age
# Question: Is churn associated with customer age?
# ============================================================

age_churn_rate = pd.crosstab(
    df["Age"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Rate by Age (%):")
print(age_churn_rate)


# ============================================================
# EDA 10: Churn by Total Spend
# Question: Is churn associated with total customer spending?
# ============================================================

df["Spend Group"] = pd.cut(
    df["Total Spend"],
    bins=10
)

spend_churn_rate = pd.crosstab(
    df["Spend Group"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Rate by Total Spend Group (%):")
print(spend_churn_rate)


# ============================================================
# EDA 11: Churn by Last Interaction
# Question: Is churn associated with last interaction?
# ============================================================

last_interaction_churn_rate = pd.crosstab(
    df["Last Interaction"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Rate by Last Interaction (%):")
print(last_interaction_churn_rate)


# ============================================================
# 5. EDA Visualization Dashboard
# All important visualizations appear in one figure.
# ============================================================

fig, axes = plt.subplots(
    3,
    3,
    figsize=(20, 15)
)

axes = axes.flatten()


# ============================================================
# Visualization 1: Churn Distribution
# ============================================================

churn_counts.plot(
    kind="bar",
    ax=axes[0]
)

axes[0].set_title("Customer Churn Distribution")
axes[0].set_xlabel("Churn")
axes[0].set_ylabel("Number of Customers")
axes[0].tick_params(axis="x", rotation=0)


# ============================================================
# Visualization 2: Churn by Contract Length
# ============================================================

contract_churn.plot(
    kind="bar",
    ax=axes[1]
)

axes[1].set_title("Churn by Contract Length")
axes[1].set_xlabel("Contract Length")
axes[1].set_ylabel("Number of Customers")
axes[1].tick_params(axis="x", rotation=0)
axes[1].legend(title="Churn")


# ============================================================
# Visualization 3: Churn Rate by Support Calls
# ============================================================

support_calls_churn_rate[1.0].plot(
    kind="line",
    marker="o",
    ax=axes[2]
)

axes[2].set_title("Churn Rate by Support Calls")
axes[2].set_xlabel("Support Calls")
axes[2].set_ylabel("Churn Rate (%)")
axes[2].set_xticks(range(11))
axes[2].grid(axis="y", alpha=0.3)


# ============================================================
# Visualization 4: Churn Rate by Payment Delay
# ============================================================

payment_delay_churn_rate[1.0].plot(
    kind="line",
    marker="o",
    ax=axes[3]
)

axes[3].set_title("Churn Rate by Payment Delay")
axes[3].set_xlabel("Payment Delay")
axes[3].set_ylabel("Churn Rate (%)")
axes[3].set_xticks(range(0, 31, 5))
axes[3].grid(axis="y", alpha=0.3)


# ============================================================
# Visualization 5: Churn Rate by Age
# ============================================================

age_churn_rate[1.0].plot(
    kind="line",
    marker="o",
    ax=axes[4]
)

axes[4].set_title("Churn Rate by Age")
axes[4].set_xlabel("Age")
axes[4].set_ylabel("Churn Rate (%)")
axes[4].set_xticks(range(18, 66, 5))
axes[4].grid(axis="y", alpha=0.3)


# ============================================================
# Visualization 6: Churn Rate by Total Spend
# ============================================================

spend_churn_rate[1.0].plot(
    kind="line",
    marker="o",
    ax=axes[5]
)

axes[5].set_title("Churn Rate by Total Spend")
axes[5].set_xlabel("Total Spend Range")
axes[5].set_ylabel("Churn Rate (%)")
axes[5].tick_params(axis="x", rotation=45)
axes[5].grid(axis="y", alpha=0.3)


# ============================================================
# Visualization 7: Churn Rate by Last Interaction
# ============================================================

last_interaction_churn_rate[1.0].plot(
    kind="line",
    marker="o",
    ax=axes[6]
)

axes[6].set_title("Churn Rate by Last Interaction")
axes[6].set_xlabel("Last Interaction")
axes[6].set_ylabel("Churn Rate (%)")
axes[6].set_xticks(range(1, 31, 3))
axes[6].grid(axis="y", alpha=0.3)


# ============================================================
# Empty panels
# ============================================================

axes[7].axis("off")
axes[8].axis("off")


# ============================================================
# Dashboard Title
# ============================================================

fig.suptitle(
    "Customer Churn - Exploratory Data Analysis",
    fontsize=20,
    fontweight="bold",
    y=0.995
)


# ============================================================
# Spacing
# ============================================================

plt.subplots_adjust(
    left=0.06,
    right=0.98,
    top=0.94,
    bottom=0.07,
    wspace=0.25,
    hspace=0.38
)


# ============================================================
# Display Dashboard
# ============================================================
# ============================================================
# Check Numerical Feature Relationships with Churn
# ============================================================

print("\nChurn Correlation:")
# Positive correlation → when the feature increases, Churn tends to increase.
# Negative correlation → when the feature increases, Churn tends to decrease.
# -1          0          +1
# │-----------│-----------│
# strong      no          strong
# negative    relationship positive
print(
    df.select_dtypes(include="number")
      .corr()["Churn"]
      .sort_values(ascending=False)
)
# Results
# Feature              Correlation       Interpretation

# Support Calls           +0.574          Stronger churn tendency
# Payment Delay           +0.312          More delay → more churn tendency
# Age                     +0.218          Moderate linear association
# Last Interaction       +0.150          Weak positive association
# Usage Frequency         -0.046          Almost no linear relationship
# Tenure                  -0.052          Almost no linear relationship
# Total Spend             -0.429          More spend → less churn tendency
# CustomerID              -0.839          Suspicious identifier relationship

# ============================================================
# Investigate Payment Delay and Churn
# ============================================================

payment_delay_churn = pd.crosstab(
    df["Payment Delay"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Rate by Payment Delay (%):")
print(payment_delay_churn)
# ============================================================
# Investigate Contract Length and Churn
# ============================================================

contract_churn_rate = pd.crosstab(
    df["Contract Length"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Rate by Contract Length (%):")
print(contract_churn_rate)
plt.show()



# ============================================================
# EDA Summary
# ============================================================

# we found patterns such as:

# Support Calls: higher calls → much higher observed churn
# Payment Delay: 21+ → 100% observed churn in this dataset
# Age: younger customers generally show higher churn
# Usage Frequency: lower usage → higher churn
# Total Spend: lower spending → higher churn
# Last Interaction: 16–30 → ~66% churn vs ~49% for 1–15
# Tenure: no clear pattern
# Subscription Type: relatively small differences
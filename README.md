# Customer Churn Prediction Using Data Mining

## Project Overview

Customer churn occurs when customers stop using a company's service or end their relationship with the company.

This project applies Data Mining classification techniques to predict whether a customer is likely to churn based on demographic, behavioral, and subscription-related information.

The project also investigates how different classification algorithms learn from the same dataset and why their performance can differ.

## Project Goal

The main goals of the project are:

1. Apply classification techniques to customer churn prediction.
2. Compare different classification algorithms.
3. Evaluate their performance using multiple metrics.
4. Investigate patterns in the data that may help explain the model results.
5. Use the selected model to predict churn for a new customer.



## Dataset

The dataset contains customer records with information such as:

- Age
- Gender
- Tenure
- Usage Frequency
- Support Calls
- Payment Delay
- Subscription Type
- Contract Length
- Total Spend
- Last Interaction

The target variable is:

- `Churn = 1` → Customer churned
- `Churn = 0` → Customer did not churn

`CustomerID` is an identifier and is not used as a predictive feature.

## Methodology

The project follows these main steps:

1. Data Understanding
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Data Preprocessing
5. Train/Test Split
6. Classification
7. Model Evaluation
8. Model Comparison
9. Model Selection
10. New Customer Prediction



### Data Preprocessing

The data is split into training and testing sets using an 80/20 split.

Categorical features are transformed using `OneHotEncoder`.

The preprocessing rules are learned only from the training data and then applied to the test data and new customers.

## Classification Models

Three classification algorithms were trained and compared:

- Logistic Regression
- Decision Tree
- Random Forest



## Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix



### Results


| Model               | Accuracy | Recall | Precision | F1-Score |
| ------------------- | -------- | ------ | --------- | -------- |
| Logistic Regression | 89.34%   | 88.54% | 92.34%    | 90.40%   |
| Decision Tree       | 99.99%   | 99.99% | 99.99%    | 99.99%   |
| Random Forest       | 99.94%   | 99.90% | 99.99%    | 99.95%   |


The Decision Tree achieved the highest performance on the test set and was selected for the final prediction stage.

## Model Interpretation

The project also investigated relationships between features and churn.

For example, the dataset showed strong observed churn patterns for:

- Payment Delay values from 21 onward
- Monthly contracts

The trained Decision Tree was also inspected to understand how it makes individual predictions.

For a sample new customer with 8 support calls, the learned tree followed this path:

```text
Support Calls > 4.5
        ↓
Support Calls > 5.5
        ↓
Churn
```

This demonstrates how a Decision Tree can transform learned patterns in the training data into explicit decision rules.

## Prediction

The trained Decision Tree and fitted preprocessing pipeline were saved and later loaded separately.

A new customer's information goes through the following process:

```text
New Customer
     ↓
Saved Preprocessor
     ↓
Processed Features
     ↓
Saved Decision Tree
     ↓
Prediction
```

The system successfully produced a churn prediction for a new customer.

## Frontend

A small Streamlit interface was added to allow users to enter customer information and receive a churn prediction through the trained Decision Tree.

## Project Structure

```text
customer-churn/
│
├── data/
│   └── customer_churn_dataset-training-master.csv/
│
├── models/
│   ├── decision_tree.pkl
│   └── preprocessor.pkl
│
├── src/
│   ├── explore_data.py
│   ├── preprocess.py
│   ├── evaluate.py
│   ├── train.py
│   ├── predict.py
│   └── app.py
│
├── README.md
└── pyproject.toml
```



### Conclusion

This project demonstrates an end-to-end Data Mining classification workflow, from data exploration and preprocessing to model training, evaluation, comparison, interpretation, and prediction on new customer data.
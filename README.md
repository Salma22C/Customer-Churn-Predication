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

The dataset contains customer records with demographic, behavioral, and subscription-related information.

### Features

- `Age`
- `Gender`
- `Tenure`
- `Usage Frequency`
- `Support Calls`
- `Payment Delay`
- `Subscription Type`
- `Contract Length`
- `Total Spend`
- `Last Interaction`

### Target Variable

- `Churn = 1` → Customer churned
- `Churn = 0` → Customer did not churn

`CustomerID` is an identifier and is not used as a predictive feature because it does not provide meaningful information for predicting churn.

### Dataset Size

The project uses the provided customer churn training dataset and an associated testing dataset.

> **Note:** The exact original dataset size should be added here based on the dataset files used in the project.

### Dataset Source

> **Dataset source:** Add the original source/link of the dataset here.

## Methodology

The project follows an end-to-end Data Mining classification workflow:

```text
Data Understanding
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis (EDA)
        ↓
Data Preprocessing
        ↓
Train/Test Split
        ↓
Classification
        ↓
Model Evaluation
        ↓
Model Comparison
        ↓
Model Selection
        ↓
New Customer Prediction
```
## Data Preprocessing

The following preprocessing steps were applied:

1. Loaded the customer churn dataset.
2. Removed rows containing missing values.
3. Removed `CustomerID` because it is an identifier rather than a predictive feature.
4. Separated the input features `X` from the target variable `y`.
5. Identified categorical and numerical features.
6. Applied One-Hot Encoding to categorical features.
7. Split the data into training and testing sets using an 80/20 split.
8. Fitted the preprocessing transformer only on the training data.
9. Applied the learned preprocessing rules to the test data.

The preprocessing process is designed to avoid data leakage:

```text
Training Data
     ↓
fit + transform
     ↓
Learn preprocessing rules
     ↓
Testing Data
     ↓
transform only
```
The same fitted preprocessor is later used when processing new customer data.
## Classification Models

Three classification algorithms were selected and compared.

### Logistic Regression

Logistic Regression was used as a classical classification baseline for the binary churn prediction problem.

### Decision Tree

Decision Tree was selected because it can learn decision rules and threshold-based relationships between customer features and churn. It also provides interpretable paths for individual predictions.

### Random Forest

Random Forest was selected as an ensemble method consisting of multiple decision trees. It provides a comparison between a single Decision Tree and an ensemble-based approach.

## Model Evaluation

The classification models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

These metrics were used to compare the performance of the three algorithms on the test set.

## Results

| Model | Accuracy | Recall | Precision | F1-Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 89.34% | 88.54% | 92.34% | 90.40% |
| Decision Tree | 99.99% | 99.99% | 99.99% | 99.99% |
| Random Forest | 99.94% | 99.90% | 99.99% | 99.95% |

The Decision Tree achieved the highest performance on the test set and was selected for the final prediction stage.

## Findings and Discussion

The three algorithms produced noticeably different results even though they were trained on the same dataset.

Logistic Regression achieved an accuracy of 89.34%, while the Decision Tree and Random Forest achieved substantially higher performance.

The dataset also contained strong observed relationships between some features and churn. For example:

- Payment Delay values from 21 onward showed 100% observed churn.
- Monthly contracts showed 100% observed churn.

These patterns can be represented naturally by tree-based models through threshold-based decision rules.

The results should nevertheless be interpreted in the context of this particular dataset and test split. The extremely high performance of the tree-based models may be related to strong patterns or separability within the dataset. Further evaluation using an independent dataset would be useful to assess how well the models generalize to unseen real-world data.

## Model Interpretation

The trained Decision Tree was inspected to understand how it makes individual predictions.

For a sample customer with 8 support calls, the learned tree followed this path:

```text
Support Calls > 4.5
        ↓
Support Calls > 5.5
        ↓
Churn
```
This demonstrates how a Decision Tree transforms patterns learned during training into explicit decision rules.

Rather than treating the model as a black box, the decision path can be inspected to understand how the prediction was reached.
## Prediction

After comparing the models, the trained Decision Tree was selected for the final prediction stage.

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

The frontend connects the user input to the same preprocessing and prediction pipeline used by the trained model.

## Project Structure

```text
customer-churn/
│
├── data/
│   ├── customer_churn_dataset-training-master.csv/
│   └── customer_churn_dataset-testing-master.csv/
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
├── pyproject.toml
└── uv.lock
```
### Conclusion

This project demonstrates an end-to-end Data Mining classification workflow for customer churn prediction.

The project covers data understanding, cleaning, exploratory analysis, preprocessing, classification, model evaluation, model comparison, model interpretation, and prediction on new customer data.

Among the three evaluated algorithms, the Decision Tree achieved the highest performance on the test set and was selected for the final prediction system.

The project also demonstrates the importance of looking beyond model scores by investigating the relationships in the data and the decision rules learned by the model.

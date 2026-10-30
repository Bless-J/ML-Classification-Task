# Telco Customer Churn Prediction

A supervised-learning project that estimates whether a telecom customer will churn. It compares interpretable Logistic Regression with a Random Forest baseline and uses a leakage-safe preprocessing and evaluation workflow.

## Project files

- [`notebooks/telco_customer_churn.ipynb`](notebooks/telco_customer_churn.ipynb): data checks, exploratory analysis, preprocessing, model comparison, and held-out evaluation.
- `requirements.txt`: Python dependencies.

## Dataset

The supplied CSV is the [Telco Customer Churn dataset on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` or rename the provided file to `Telecom_Churn.csv`, then place it in `data/` or `/content/` in Colab. The data is not committed here; review the Kaggle source terms before reuse.

The provided file contains 7,043 customers and 21 columns. Churn is 26.5% of the rows (1,869 customers). `TotalCharges` has 11 blank values, which are treated as missing and imputed inside the training pipeline.

## Run

1. Install dependencies: `pip install -r requirements.txt`.
2. Download the dataset from Kaggle and put the CSV in `data/` or `/content/`.
3. Open `notebooks/telco_customer_churn.ipynb` in Jupyter or Google Colab and run all cells.

## Method

The target is `Churn` (`Yes` = 1). The customer identifier is dropped. A stratified train/test split is created before any imputation, encoding, or scaling. Preprocessing is fitted only on the training portion through scikit-learn pipelines. Logistic Regression and Random Forest are compared using accuracy, precision, recall, F1, ROC-AUC, average precision, and a confusion matrix. Class weighting gives the models an imbalance-aware baseline without generating synthetic customers.

## Review notes

The original notebook split the dataframe, then built both train and test features from the full dataset. As a result, the evaluation was not held out and preprocessing/model selection could leak information. This cleaned notebook fixes that design. The earlier model scores are not carried forward; run the corrected notebook to generate valid results.

## Limitations

This is a public sample dataset, not a live telecom feed. It supports a classification exercise, not a retention intervention claim. Model scores do not demonstrate business impact, and the cost of false positives versus missed churners should inform threshold selection before operational use.

## LinkedIn project description

**Telecom Customer Churn Prediction** — Built a leakage-safe classification workflow for customer churn using pandas and scikit-learn. Compared Logistic Regression and Random Forest with training-only preprocessing, stratified evaluation, and churn-focused metrics, while documenting class imbalance and the limits of translating model scores into retention impact.

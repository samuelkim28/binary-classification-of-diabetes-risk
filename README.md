# Binary Classification of Diabetes Risk Using Machine Learning

## Table of Contents

- [Project Overview](#project-overview)
- [Data](#data)
- [Methodology](#methodology)
- [Installation](#installation)
- [Results](#results)

---

## Project Overview

**Goals:**
- Predict the presence of diabetes using clinical health indicators, emphasizing model trade-offs and minimizing false negatives.
- Gain experience building end-to-end machine learning workflows, including data preprocessing, model training, evaluation using multiple metrics, and result interpretation.

**Dataset:** See the [Data](#data) section for details.

**Key Challenges:**
- Class imbalance
- High cost of false negatives in healthcare settings

---

## Data

- **Source:** [Pima Indians Diabetes Dataset (Kaggle)](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database/data)
- **Samples:** 768 patients
- **Features:** 8 clinical health indicators (e.g., glucose, BMI, age)
- **Target:** Binary label (`1` = diabetes, `0` = no diabetes)
- **Notes:** Dataset contains no missing values; all features are numeric

---

## Methodology

The project follows a structured machine learning workflow:

1. **Data Acquisition and Exploration**: Imported the Pima Indians Diabetes dataset from Kaggle and conducted an initial review of the data to understand its structure, feature types, and target variable. This step helped identify the key predictors and ensured the dataset was ready for modeling.
2. **Data Preprocessing**: Split the dataset into training and testing sets. No missing values were present, so no imputation was needed; all features are numeric.  
3. **Model Selection and Training**: Implemented multiple classification algorithms, including K-Nearest Neighbors (KNN), Decision Tree, Random Forest, Logistic Regression, and Support Vector Machine (SVM). Each model was trained on the training set.  
4. **Evaluation and Visualization**: Evaluated model performance using accuracy, precision, recall, and F1-score. Results were visualized with plots to compare model performance.  
5. **Model Focus and Hyperparameter Tuning**: Focused on Random Forest as it achieved the best performance. Conducted parameter experiments (e.g., number of estimators, max depth) to optimize performance, with results visualized for clarity.

---

## Installation

This project is designed to run in **Google Colab**, so no local setup is required. To get started:

1. Ensure you have a Google account.
2. Open the notebook in Google Colab.

All necessary Python libraries (e.g., `pandas`, `scikit-learn`, `matplotlib`) are imported within the notebook itself.

---

## Results

Multiple classification models were evaluated, including KNN, Random Forest, Decision Tree, Logistic Regression, and SVM.    
Among these, the **Random Forest classifier (entropy criterion)** achieved the strongest overall performance and was selected for further analysis.

**Random Forest Performance (Test Set):**
- **Accuracy:** 0.84
- **Precision:** 0.83
- **Recall:** 0.69
- **F1-score:** 0.75

The confusion matrix below summarizes prediction outcomes:

- True Negatives: 91  
- False Positives: 8  
- False Negatives: 17  
- True Positives: 38  

The model demonstrates strong overall accuracy and precision, while recall is comparatively lower, reflecting the inherent trade-off between minimizing false positives and false negatives in a healthcare context. This result highlights the importance of recall when predicting diabetes risk and motivates further tuning or alternative approaches to reduce false negatives.

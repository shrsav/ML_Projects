# ML_Project 
Machine Learning Project 1  

# Customer Churn Prediction & Model Comparison

An end-to-end machine learning project designed to predict customer churn for a telecommunications provider. This project evaluates multiple classification models—Logistic Regression, Random Forest, and XGBoost—to identify high-risk customers and extract key business drivers behind customer attrition.

## Project Overview

Customer churn poses a significant financial challenge in subscription-based business models. Acquiring new customers is often significantly more expensive than retaining existing ones. 

The primary goal of this project is to:
1. Build a machine learning pipeline to accurately detect potential churners.
2. Compare linear and ensemble classifiers to find the best model for imbalanced churn data.
3. Extract and visualize key feature importances to provide actionable business recommendations.

---

## Dataset & Preprocessing

The dataset contains customer demographic data, account details, and service usage information.

**Preprocessing Steps:**
* **Data Cleaning:** Converted `TotalCharges` to numeric values, coercing invalid entries and handling missing data safely.
* **Feature Dropping:** Removed non-predictive identifiers (`customerID`).
* **Binary Mapping:** Converted the target variable (`Churn`) into binary indicators (`1` for Churned, `0` for Retained).
* **Categorical Encoding:** Applied one-hot encoding (`pd.get_dummies` with `drop_first=True`) to handle categorical features while avoiding the dummy variable trap.
* **Data Splitting:** Applied an 80/20 train-test split (`random_state=42`).

---

## Models & Performance Comparison

Three algorithms were trained and evaluated on the test set:

| Model | Accuracy | Churned Precision | Churned Recall | Churned F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | **82%** | **0.69** | **0.60** | **0.64** |
| **XGBoost** | 80% | 0.64 | 0.53 | 0.58 |
| **Random Forest** | 79% | 0.65 | 0.45 | 0.53 |

### Key Takeaways
* **Logistic Regression** achieved the highest overall accuracy (**82%**) and the strongest recall for churned customers (**60%**).
* Simple linear boundaries outperformed complex tree ensembles (Random Forest and XGBoost) out-of-the-box on this dataset, proving particularly effective at capturing the minority class.

---

## Feature Importance Analysis

Using **XGBoost**, the top 10 most influential features driving customer retention and churn were extracted and visualized. 

The saved visualization (`feature_importances.png`) highlights how specific contract types, tenure duration, and billing methods impact customer decisions.

---

## Tech Stack & Dependencies

* **Language:** Python 3.x
* **Data Handling:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn`, `xgboost`
* **Visualization:** `matplotlib`

---

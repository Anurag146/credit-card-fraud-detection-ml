# Credit Card Fraud Detection: A Machine Learning Approach

## Project Overview
This project addresses the challenge of identifying fraudulent credit card transactions in a highly imbalanced dataset. Financial fraud detection is a "needle in a haystack" problem where missing a single transaction (False Negative) can be far more costly than a false alarm. I implemented a robust ML pipeline that prioritizes high **Recall** to ensure security while maintaining **Precision** to protect the user experience.

## Dataset Information
The project utilizes a dataset containing transactions made by European cardholders. 

- **Source:** [Download Dataset (creditcard.csv)](https://drive.google.com/file/d/1Fr9YUtqCBozg_RFqptGYtBG4lajD_jS-/view?usp=sharing)
- **Total Transactions:** 284,807
- **Class Distribution:** 284,315 normal (0), 492 fraudulent (1)
- **Imbalance Ratio:** 0.17% (Fraud)

## Tech Stack & Tools
- **Language:** Python 3.9+
- **Libraries:** Pandas, Scikit-Learn, Imbalanced-Learn (SMOTE), Matplotlib, Seaborn, Joblib
- **Environment:** VS Code, Jupyter Notebooks

## Key Features of the Pipeline
- **Exploratory Data Analysis (EDA):** Identified that fraud transactions often cluster at lower amounts to avoid detection.
- **Advanced Preprocessing:** Applied `StandardScaler` to 'Time' and 'Amount' and handled severe class imbalance using **SMOTE**.
- **Model Optimization:** Iterated from a baseline Logistic Regression to an optimized **Random Forest Classifier**.
- **Production Readiness:** Modularized the code into a Python class structure with logging and model serialization.

## Results & Performance
| Metric | Logistic Regression | Random Forest (Final) |
| :--- | :--- | :--- |
| **Recall (Fraud)** | 92% | 82% |
| **Precision (Fraud)** | 6% | 87% |
| **F1-Score** | 0.11 | 0.84 |

### Feature Importance
The model identified **V17, V14, and V12** as the primary indicators of fraudulent behavior, suggesting that latent behavioral patterns are more predictive than transaction size alone.

![Feature Importance](./visuals/feature_importance.png)

## Acknowledgment
This project was completed as part of my Data Science Internship with **@Skillfied Mentor**.

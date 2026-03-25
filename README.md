# Credit Card Fraud Detection: A Machine Learning Approach

## Project Overview
This project addresses the challenge of identifying fraudulent credit card transactions in a highly imbalanced dataset (0.17% fraud). I implemented a robust ML pipeline that prioritizes high recall to ensure financial security while maintaining precision to protect user experience.

## Tech Stack & Tools
- **Language:** Python 3.9+
- **Libraries:** Pandas, Scikit-Learn, Imbalanced-Learn (SMOTE), Matplotlib, Seaborn
- **Environment:** VS Code, Jupyter Notebooks

## Key Features
- **Exploratory Data Analysis (EDA):** Visualized transaction patterns and class distributions.
- **Advanced Preprocessing:** Scaled transactional features and handled severe class imbalance using SMOTE.
- **Model Optimization:** Compared Logistic Regression against Random Forest, achieving a balanced F1-score.

## Results & Performance
| Metric | Logistic Regression | Random Forest (Final) |
| :--- | :--- | :--- |
| **Recall (Fraud)** | 92% | 82% |
| **Precision (Fraud)** | 6% | 87% |

### Feature Importance
The model identified **V17, V14, and V12** as the primary indicators of fraudulent behavior, suggesting that latent behavioral patterns are more predictive than transaction size alone.

![Feature Importance](./visuals/feature_importance.png)

## Repository Structure
- `notebooks/`: Contains the step-by-step experimentation and EDA.
- `visuals/`: High-resolution charts and confusion matrices.
- `data/`: Dataset link and metadata.

## Acknowledgment
This project was completed as part of my Data Science Internship with **@Skillfied**.

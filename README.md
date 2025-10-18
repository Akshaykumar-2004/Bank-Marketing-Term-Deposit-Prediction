# Bank Marketing Term Deposit Prediction

## Overview
This project analyzes the **Bank Marketing (UCI)** dataset to identify factors influencing customer decisions to subscribe to a term deposit during a phone marketing campaign. The goal is to provide the bank with actionable insights to improve campaign efficiency and targeting, ultimately increasing the conversion rate. Additionally, a **K-Nearest Neighbors (KNN)** classification model is built to predict subscription likelihood.

## Dataset
The analysis uses the `bank-additional-full.csv` file from the **Bank Marketing (UCI)** dataset available on Kaggle. This dataset contains information on over **41,000 marketing calls**, including customer demographics, socio-economic factors, previous campaign interactions, and the final outcome (subscribed 'yes' or 'no').

## Methodology

### 1. Data Loading & Initial EDA (Python - Google Colab)
- Loaded the dataset using **Pandas**, specifying the semicolon separator.
- Inspected the data (`.head()`, `.info()`) and confirmed no missing values (`.isnull().sum()`).
- Calculated and visualized the overall conversion rate (~11.3%).
- Analyzed demographics (Job, Marital Status, Education, Age) using Seaborn visualizations.
- Investigated the impact of existing debt (housing, personal loans) on subscription rates.

### 2. Advanced EDA, Cleaning & Feature Engineering
- Identified and capped outliers in the `duration` column using the **IQR method**.
- Engineered a new feature `age_group` (e.g., ‘Young Adult’, ‘Adult’) using `pd.cut` for improved interpretability.
- Analyzed campaign effectiveness (`campaign` variable) and macroeconomic indicators (`euribor3m`) via boxplots.

### 3. Data Simplification (PCA)
- Converted categorical variables to numerical via `pd.get_dummies`.
- Scaled features with **StandardScaler** to prepare for PCA.
- Applied **Principal Component Analysis (PCA)** to reduce 50+ features to 10 components explaining ~50% variance.

### 4. Predictive Modeling (KNN)
- Split PCA-transformed data into **70% training / 30% testing** sets.
- Trained a **K-Nearest Neighbors Classifier (K=5)**.
- Evaluated performance with **Accuracy**, **Confusion Matrix**, and **Classification Report (Precision, Recall, F1-Score)**.

## Key Insights

| Question | Insight |
|-----------|----------|
| **Overall Success Rate** | Conversion rate is ~11.3%, showing low campaign efficiency. |
| **Target Audience** | Students and retirees are more likely to subscribe than admin or blue-collar workers. |
| **Debt Dilemma** | Customers without housing or personal loans are more likely to subscribe. |
| **Outlier Calls** | Extreme `duration` values were detected and capped. |
| **Age Grouping** | 'Young Adult' and 'Elder' age groups show the highest subscription rates. |
| **Contact Strategy** | 1–2 calls are most effective; excessive calls lead to diminishing returns. |
| **Economic Impact** | Lower `euribor3m` interest rates correlate with higher subscriptions. |
| **Simplification** | PCA effectively reduced dimensionality while retaining interpretive power. |
| **Prediction** | KNN achieved ~90% accuracy, though recall for “yes” was low — better at predicting “no”. |

## Technologies Used
- **Python**
- **Pandas**, **NumPy**
- **Matplotlib**, **Seaborn**
- **Scikit-learn** (StandardScaler, PCA, train_test_split, KNeighborsClassifier, metrics)
- **SQLite** (for optional data storage)
- **Google Colab**

---

### 📊 Outcome
This project provided clear evidence of which customer profiles and economic conditions drive term deposit subscriptions. The findings enable data-driven targeting strategies and lay groundwork for advanced model tuning (e.g., SMOTE for class imbalance or grid search for hyperparameter optimization).

---

### 📁 File Outputs
- `cleaned_bank_data.csv` – post-EDA cleaned dataset  
- `pca_transformed_data.csv` – PCA-reduced dataset  
- `model_results.txt` – KNN performance metrics  

---

**Author:** *[Your Name]*  
**Dataset Source:** [UCI Bank Marketing Dataset on Kaggle](https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing)  

#  Online Transaction Fraud Detection

This project analyzes and models fraudulent online transactions using machine learning techniques. It includes data preprocessing, exploratory data analysis (EDA), feature engineering, and classification model evaluation using metrics like ROC-AUC.

---

##  Project Structure 
 online-transaction-fraud-project 
-  project03-18-2025.py # Main Python script for processing and modeling 
-  onlinefraud.csv # Dataset (free online from Kaggle.com) 
-  weight_feature_importance_online_fraud.png # XGBoost importance by weight 
-  gain_feature_importance_online_fraud.png # XGBoost importance by gain 
-  cover_feature_importance_online_fraud.png # XGBoost importance by cover 
-  Figure_6.png # Confusion matrix (Random Forest)
-  README.md # Project overview


---

##  Dataset

The dataset contains transaction records with the following features:

- `step`: Time step (1 = 1 hour)
- `type`: Type of transaction (e.g., CASH_OUT, PAYMENT)
- `amount`: Transaction amount
- `oldbalanceOrg` / `newbalanceOrig`: Sender's balance before/after
- `oldbalanceDest` / `newbalanceDest`: Receiver's balance before/after
- `isFraud`: 1 if the transaction is fraudulent, 0 otherwise

New engineered features:
- `balanceDiffOrig`: Difference in sender’s balance
- `balanceDiffNew`: Difference in receiver’s balance

---

##  Methodology

- Cleaned and prepared data (no null values)
- Encoded categorical features using `OrdinalEncoder`
- Engineered new features for deeper pattern detection
- Split into train/test using `train_test_split`

###  Models Used:

- **Random Forest Classifier**
- **XGBoost Classifier** (used for feature importance visualization)

---

##  Key Visualizations

###  Transaction Type Distribution

![Transaction Types](https://github.com/nkerudi/online-transaction-fraud-project/blob/main/Figure_6.png)

###  XGBoost Feature Importances

- **By Weight**  
  ![Feature Importance - Weight](https://github.com/nkerudi/online-transaction-fraud-project/blob/main/weight_feature_importance_online_fraud.png)

- **By Gain**  
  ![Feature Importance - Gain](https://github.com/nkerudi/online-transaction-fraud-project/blob/main/gain_feature_importance_online_fraud.png)

- **By Cover**  
  ![Feature Importance - Cover](https://github.com/nkerudi/online-transaction-fraud-project/blob/main/cover_feature_importance_online_fraud.png)

---

##  Results

**Random Forest Results:**
- Confusion Matrix:  
  True Positives (Fraud correctly predicted): **1,914**  
  False Negatives (Fraud missed): **521**

- Overall Accuracy: (printed in console via `accuracy_score`)

- Strong model performance despite class imbalance

---

##  How to Run

1. Clone this repo:
   ```bash
   git clone https://github.com/nkerudi/online-transaction-fraud-project.git
   cd online-transaction-fraud-project




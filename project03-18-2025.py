import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, roc_curve


#classifying online transactions as fraudulent or non-fraudulent 
#what types of transactions lead to fraud 
#columns in data set:
#step: represents unit of time where one setp is 1 hour 
#type: type of online transaction 
#amount: amount of transaction 
#nameOrig: customer starting transaction 
#oldBalanceOrg: balance before transaction 
#newBalanceOrig: balance after transaction 
#nameDest: recipient of transaction 
#oldBalanceDest: initial balance of recipient before transaction 
#newBalanceDest: balance of recipient after transaction 
#isFraud: fraud transaction/not 

#XGBoost: 
#boosting (builds an ensemble of decision trees sequenstially 
# each tree corrects the errors from the last one, resulting in a stronger 
#predictive model)

#1. Loading and Exploring Data 
df = pd.read_csv('onlinefraud.csv')
df.describe()
df.info()
df['balanceDiffOrig'] = df['newbalanceOrig'] - df['oldbalanceOrg']
df['balanceDiffNew'] = df['newbalanceDest'] - df['oldbalanceDest']
df.head()
df.isnull().sum() #shows no null values 

#2. Data Preprossessing (handle missing vals, split data)
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, MinMaxScaler 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
encoder = OrdinalEncoder()
df['type'] = encoder.fit_transform(df['type'])
X = df.drop('isFraud', axis = 1)
print(X)
y = df['isFraud']
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state= 42)
model = RandomForestClassifier(n_estimators = 100, max_depth = 5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")


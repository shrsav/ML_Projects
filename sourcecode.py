import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier 


df = pd.read_csv("data.csv")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].str.strip(), errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)
df = df.drop(columns=['customerID'])

df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
X = pd.get_dummies(df.drop(columns=['Churn']), drop_first=True)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

logreg = LogisticRegression(max_iter = 1000)
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)
print(classification_report(y_test, y_pred_logreg, target_names=['Retained', 'Churned']))
print(y_pred_logreg)

randomforest = RandomForestClassifier(n_estimators = 150, random_state= 42)
randomforest.fit(X_train, y_train)
y_pred_rf = randomforest.predict(X_test)
print(classification_report(y_test, y_pred_rf, target_names=['Retained', 'Churned']))

xgbclass = XGBClassifier(random_state=42, eval_metric='logloss')
xgbclass.fit(X_train, y_train)
y_pred_xgb = xgbclass.predict(X_test)
print(classification_report(y_test, y_pred_xgb, target_names=['Retained', 'Churned']))

acclog = accuracy_score(y_test, y_pred_logreg, normalize=True, sample_weight=None)
print(acclog)
accfor = accuracy_score(y_test, y_pred_rf, normalize=True, sample_weight=None)
print(accfor)
accxgb = accuracy_score(y_test, y_pred_xgb, normalize=True, sample_weight=None)
print(accxgb)

importances = pd.Series(xgbclass.feature_importances_, index=X.columns)
plt.figure(figsize=(8, 5))
importances.nlargest(10).sort_values().plot(kind='barh')
plt.title("Top 10 Feature Importances using XGBoost")
plt.xlabel("Relative Importance")
plt.tight_layout()
plt.savefig("feature_importances.png")
plt.show()

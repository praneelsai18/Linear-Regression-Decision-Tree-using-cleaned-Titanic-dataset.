import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import joblib
import os

data = {
    'PassengerId': [1, 2, 3, 4, 5, 6],
    'Sex': ['male', 'female', 'female', 'male', 'male', 'male'],
    'Age': [22, 38, 26, 35, 32, 54],
    'SibSp': [1, 1, 0, 0, 0, 0],
    'Parch': [0, 0, 0, 0, 0, 0],
    'Fare': [7.25, 71.28, 7.92, 8.05, 8.46, 51.86],
    'Embarked': ['S', 'C', 'S', 'S', 'S', 'S'],
    'Survived': [0, 1, 1, 0, 0, 1]
}

df = pd.DataFrame(data)

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

print("✅ Titanic Dataset Created Successfully!")
print(df)

print("\n🚀 Training Linear Regression Model to Predict Fare...")

X_reg = df[['PassengerId', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']]
y_reg = df['Fare']

X_train, X_test, y_train, y_test = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_pred_reg = lr_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred_reg)
r2 = r2_score(y_test, y_pred_reg)

print("\n📊 Linear Regression Evaluation")
print(f"MSE: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

print("\n🌳 Training Decision Tree Classifier to Predict Survival...")

X_clf = df[['PassengerId', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y_clf = df['Survived']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_clf, y_clf, test_size=0.3, random_state=42
)

dt_model = DecisionTreeClassifier()
dt_model.fit(X_train_c, y_train_c)

y_pred_c = dt_model.predict(X_test_c)

accuracy = accuracy_score(y_test_c, y_pred_c)

print("\n📊 Decision Tree Evaluation")
print(f"Accuracy: {accuracy:.4f}")

os.makedirs("model", exist_ok=True)

joblib.dump(lr_model, "model/linear_regression_titanic.joblib")
joblib.dump(dt_model, "model/decision_tree_titanic.joblib")

print("\n💾 Models saved in /model folder!")
print("\n✅ Mini Project 4 Completed Successfully!")
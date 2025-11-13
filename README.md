# Linear Regression & Decision Tree using Titanic Dataset (Standalone)

## Overview
This mini project demonstrates the application of **supervised machine learning models** using a **standalone Titanic dataset**.  
The project trains:
-  **Linear Regression** (Regression Task — Predict Fare)
-  **Decision Tree Classifier** (Classification Task — Predict Survival)

This project is **self-contained** and does **not depend** on the output of Mini Project 3.  
A small Titanic-like dataset is created directly inside the Python script.

---

##  Objectives
###  Data Preparation
- Create Titanic dataset (Passenger features + Survived label)
- Encode categorical features into numeric form  
- Prepare data for ML models

###  Model Training
- Train Linear Regression model for predicting **Fare**
- Train Decision Tree Classifier for predicting **Survival**

###  Model Evaluation
- Evaluate Linear Regression using:
  - Mean Squared Error (MSE)
  - R² Score

- Evaluate Decision Tree Classifier using:
  - Accuracy

###  Save Models
Both models are saved in the `model/` folder using **joblib**.

---

##  Project Structure

mini_project_4_titanic_ml/ │ ├── main.py           
main ML code ├── requirements.txt     
dependencies └── model/         
saved ML models (auto-created)

---

##  Installation

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt

### Step 2 — Run the project
python main.py
---

Machine Learning Models Used

1. Linear Regression (Regression)

Goal: Predict the passenger Fare
Features used:
PassengerId
Sex
Age
SibSp
Parch
Embarked

Evaluation Metrics:
MSE (Mean Squared Error)
R² Score
---

2. Decision Tree Classifier (Classification)

Goal: Predict whether the passenger Survived (0 = No, 1 = Yes)
Features used:
PassengerId
Sex
Age
SibSp
Parch
Fare
Embarked

Evaluation Metric:
Accuracy

---

Sample Output (Example)

Titanic Dataset Created Successfully!
Training Linear Regression Model...
MSE: 14.3210
R² Score: 0.8123
Training Decision Tree Classifier...
Accuracy: 0.6667
Models saved in /model folder!
Completed Successfully!


---

Saved Models
Models are stored automatically when you run the script:
model/
│
├── linear_regression_titanic.joblib
├── decision_tree_titanic.joblib


---

Conclusion
This project successfully demonstrates:
Preparation of a standalone dataset
Implementation of supervised machine learning
Regression (predicting Fare)
Classification (predicting Survival)
Evaluation using appropriate metrics
It provides a clear understanding of how to train, test, and evaluate ML models using Scikit-Learn.

Author
G. Sai Praneel
Department of Computer Science & Engineering

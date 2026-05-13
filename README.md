# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import required libraries.
2. Load the dataset using pandas.
3. Create a copy of the dataset.
4. Remove unnecessary columns (sl_no, salary).
5. Check for null values and duplicates.
6. Convert categorical data into numerical data using Label Encoding.
7. Separate input features (x) and target variable (y).
8. Split the dataset into training and testing data.
9. Create the Logistic Regression model.
10. Train the model using training data.
11. Predict output using test data.
12. Calculate accuracy of the model.
13. Generate classification report.
14. Predict placement status for new student data.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: cholimgapuram sai likitha
RegisterNumber:  212224230046
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv('Placement_Data (1).csv')
data = data.drop('sl_no', axis=1)
data = data.drop('salary', axis=1)
data["gender"] = data["gender"].astype('category')
data["ssc_b"] = data["ssc_b"].astype('category')
data["hsc_b"] = data["hsc_b"].astype('category')
data["degree_t"] = data["degree_t"].astype('category')
data["workex"] = data["workex"].astype('category')
data["specialisation"] = data["specialisation"].astype('category')
data["status"] = data["status"].astype('category')
data["hsc_s"] = data["hsc_s"].astype('category')
data["gender"] = data["gender"].cat.codes
data["ssc_b"] = data["ssc_b"].cat.codes
data["hsc_b"] = data["hsc_b"].cat.codes
data["degree_t"] = data["degree_t"].cat.codes
data["workex"] = data["workex"].cat.codes
data["specialisation"] = data["specialisation"].cat.codes
data["status"] = data["status"].cat.codes
data["hsc_s"] = data["hsc_s"].cat.codes
print(data)
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
theta = np.random.randn(x.shape[1])
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def loss(theta, X, y):
    h = sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
def gradient_descent(theta, X, y, alpha, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h - y) / m
        theta -= alpha * gradient
    return theta
theta = gradient_descent(theta, x, y, alpha=0.01, num_iterations=1000)
def predict(theta, X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h >= 0.5, 1, 0)
    return y_pred
y_pred = predict(theta, x)
accuracy = np.mean(y_pred.flatten() == y)
print("Accuracy:", accuracy)
print("Predicted Values:")
print(y_pred)
xnew = np.array([[0, 87, 0, 95, 0, 2, 78, 2, 0, 0, 1, 0]])
y_prednew = predict(theta, xnew)
print("New Prediction 1:")
print(y_prednew)
xnew = np.array([[0, 0, 0, 0, 0, 2, 8, 2, 0, 0, 1, 0]])
y_prednew = predict(theta, xnew)
print("New Prediction 2:")
print(y_prednew)
*/
```

## Output:
<img width="757" height="535" alt="image" src="https://github.com/user-attachments/assets/7aff0e53-5413-4395-8630-8f826611c07b" />



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.


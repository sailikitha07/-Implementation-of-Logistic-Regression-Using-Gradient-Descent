# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import required libraries.
2. Load the dataset using read_csv().
3. Convert target column (status) into numeric values:
4. Placed → 1
5. Not Placed → 0
6. Select input features (ssc_p, mba_p) as X.
7. Select output column (status) as y.
8. Normalize input data using StandardScaler().
9. Add bias column (ones) to input data.
10. Define the sigmoid activation function.
11. Define the cost function for logistic regression.
12. Initialize weights (theta) with zeros.
13. Set learning rate and number of iterations.
14. Repeat Gradient Descent process:
15. Calculate predicted values
16. Compute gradients
17. Update weights
18. Calculate and store cost
19. Predict final output values using sigmoid function.
20. Convert probabilities into class labels (0 or 1).
21. Calculate model accuracy.
22. Print weights and accuracy.
23. Plot graph between iterations and cost function.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: cholimgapuram sai likitha
RegisterNumber:  212224230046
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("Placement_Data (2).csv")

data['status'] = data['status'].map({'Placed': 1, 'Not Placed': 0})

X = data[['ssc_p', 'mba_p']].values
y = data['status'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

m = len(y)
X = np.c_[np.ones(m), X]

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost_function(X, y, theta):
    h = sigmoid(X @ theta)
    return (-1/m) * np.sum(y*np.log(h) + (1-y)*np.log(1-h))

theta = np.zeros(X.shape[1])
alpha = 0.1
cost_history = []

for i in range(500):
    z = X @ theta
    h = sigmoid(z)
    gradient = (1/m) * X.T @ (h - y)
    theta = theta - alpha * gradient
    
    cost = cost_function(X, y, theta)
    cost_history.append(cost)

y_pred = (sigmoid(X @ theta) >= 0.5).astype(int)

accuracy = np.mean(y_pred == y) * 100
print("Weights:", theta)
print("Accuracy:", accuracy, "%")

plt.figure()
plt.plot(cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Logistic Regression using Gradient Descent")
plt.show()
*/
```

## Output:
<img width="927" height="997" alt="image" src="https://github.com/user-attachments/assets/d5fd0c96-1d9b-489c-89d0-3527b2e72814" />



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.


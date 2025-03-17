# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations of gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph.  

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: TRISHA PRIYADARSHNI PARIDA
RegisterNumber:  212224230293
*/

from google.colab import drive
drive.mount('/content/drive')

#importing necessary libraries and modules
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

#importing "50_Startups.csv" file given to work with
from google.colab import files
uploaded = files.upload()
data = pd.read_csv('/content/50_Startups.csv')
def linear_regression(X1,y,learning_rate = 0.1, num_iters = 1000):

# Add a column of ones to x for intercept term
    X = np.c_[np.ones(len(X1)),X1]

# Initialise theta with zeroes
    theta = np.zeros(X.shape[1]).reshape(-1,1)

# Perform gradient descent
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1,1)

# Calculate errors
        errors=(predictions - y ).reshape(-1,1)

# Update theta using gradient descent
        theta -= learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta

# Assuming the last column is your target variable 'y' and the preceding columns are your features 'x'
data.head()
X=(data.iloc[1:,:-2].values)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
print(X1_Scaled)

# Learn Model Parameters

theta= linear_regression(X1_Scaled,Y1_Scaled)

# Predict data value for a new data point

new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")
```

## Output:

![Screenshot 2025-03-17 212925](https://github.com/user-attachments/assets/6025c3b5-93a5-42ef-b62c-9200ce291e38)
![Screenshot 2025-03-17 212935](https://github.com/user-attachments/assets/6bc78dcd-f704-4c41-86e6-9204efdbcfd8)
![Screenshot 2025-03-17 212941](https://github.com/user-attachments/assets/51543e92-c0f7-4e7e-a402-449ffaab8670)
![Screenshot 2025-03-17 212949](https://github.com/user-attachments/assets/4fe7fc9f-67d5-4ffd-bec1-d0a8d15790e6)
![Screenshot 2025-03-17 213006](https://github.com/user-attachments/assets/20fce3ff-fc64-4652-8721-309fae226a0d)





## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.

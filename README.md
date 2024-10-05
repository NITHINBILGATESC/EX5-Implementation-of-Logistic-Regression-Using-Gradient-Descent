# EX 5 Implementation of Logistic Regression Using Gradient Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.
## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook
## Algorithm
1.Initialize the weight vector to zeros or small random values.
2.The number of times the algorithm will run through the entire dataset. 
3.The sigmoid function maps any real-valued number into the (0, 1) interval, which is useful for binary classification.
4. For each iteration, perform the following steps

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by:NITHIN BILGATES C 
RegisterNumber:2305001022
*/
import pandas as pd
import numpy as np
d=pd.read_csv("/ex45Placement_Data (3).csv")
d.head()
data1=d.copy()
data1.head()
data1=data1.drop(['sl_no','salary'],axis=1)
data1
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
X=data1.iloc[:,:-1]
Y=data1["status"]
theta=np.random.randn(X.shape[1])
y=Y
def sigmoid(z):
  return 1/(1+np.exp(-z))
def loss(theta,X,y):
  h=sigmoid(X.dot(theta))
  return -np.sum(y*np.log(h)+(1-y)*log(1-h))
def gradient_descent(theta,X,y,alpha,num_iterations):
  m=len(y)
  for i in range(num_iterations):
    h=sigmoid(X.dot(theta))
    gradient=X.T.dot(h-y)/m
    theta-=alpha*gradient
  return theta
theta=gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)
def predict(theta,X):
  h=sigmoid(X.dot(theta))
  y_pred=np.where(h>=0.5,1,0)
  return y_pred
y_pred=predict(theta,X)
accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy)
print("Predicted:\n",y_pred)
print("Actual:\n",y.values)
xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print("Predicted Result:",y_prednew)
## Output:
![image](https://github.com/user-attachments/assets/ec19495f-e449-4782-bf29-6757074f74d7)
![image](https://github.com/user-attachments/assets/982d296d-d34f-44f0-bbcb-892c530f6f24)
![image](https://github.com/user-attachments/assets/eb896771-0829-4d37-84de-533ccad06395)
![image](https://github.com/user-attachments/assets/e22e50d8-51ca-4077-9391-b9f80ac825da)
![image](https://github.com/user-attachments/assets/b02d2349-f147-4616-a38f-32f4974a790a)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.


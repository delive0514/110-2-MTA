from hmac import trans_36
import pandas as pd 
import numpy as np
import numpy.linalg
import math
from sklearn import *
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.metrics import confusion_matrix


x = open("iris_x.txt", "r")
y = open("iris_y.txt", "r")
data =[]
temp =[]
for i in x:
    data.append(np.array(i.split(), dtype=float))
dataSeries = np.array(data)
# print(dataSeries)

for i in y:
    temp.append(np.array(i.split(), dtype=int))
label = np.array(temp)
# print(label)

train_x, test_x, train_y, test_y = train_test_split(dataSeries, label, random_state=20220413, train_size=0.8)

reg = linear_model.LinearRegression()
reg.fit(train_x, train_y)
MSE = mean_squared_error(test_y, reg.predict(test_x))
print("MSE= ",MSE)

class QD():
    def ___init__(self):
        self.mu=np.array([])
        self.cov=np.array([])
    def fit(self, data, label):
        mu, cov=[],[]
        for i in range(np.max(label)+1):
            pos = np.where(label==i)[0]
            tmp_data = data[pos,:]
            tmp_cov = np.cov(np.transpose(tmp_data))
            tmp_mu = np.mean(tmp_data,axis=0)
            mu.append(tmp_mu)
            cov.append(tmp_cov)
        self.mu = np.array(mu)
        self.cov = np.array(cov)

    def predict(self,test_x):
        result =[]
        for test in test_x:
            d_value=[]
            for tmp_mu, tmp_cov in zip(self.mu, self.cov):
                zero_center_data = test - tmp_mu
                tmp = np.dot(np.transpose(zero_center_data), np.linalg.inv(tmp_cov))
                tmp = -0.5*np.dot(tmp, zero_center_data)
                tmp1= -0.5* math.log(np.linalg.norm(tmp_cov))
                tmp1 = tmp + tmp1
                d_value.append(tmp1)
            d_value = np.array(d_value)
            result.append(np.argmax(d_value))
        return result

    def score(self, y_data, y_predict) -> float:
        correct = 0
        for i in range(len(y_predict)):
            if(y_predict[i] == y_data[i]):
                correct += 1
        return correct / len(y_data)

    def confusion_matrix(y_data, y_predict) -> np.array:
        n = len(np.unique(y_data))
        result = np.empty((n, n), dtype=np.uint)
        result.fill(0)
        for i in range(len(y_predict)):
            result[y_data[i]][y_predict[i]] += 1
        return result

        
qda = QD()
qda.fit(train_x,train_y)
result = qda.predict(test_x)
Accuracy = qda.score(test_y, result)

print("acc = %.3f%%" % (float(Accuracy) * 100))
print(f"confusion_matrix: \n{confusion_matrix(test_y, result)}")


qda = QDA()
qda.fit(train_x, train_y)

# 預判
predicted = qda.predict(test_x)

# 得到準確率
Accuracy = qda.score(test_x, test_y)

# 輸出結果
print("Accurate = %.3f%%" % (float(Accuracy) * 100))
print(f"Confusion Matrix: \n{confusion_matrix(test_y, predicted)}")


x.close()
y.close()
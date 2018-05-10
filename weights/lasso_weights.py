#!/usr/bin/env python 
from __future__ import print_function

import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np  
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score 
from sklearn.linear_model import Lasso 
import pandas as pd 
import argparse 

parser= argparse.ArgumentParser()
parser.add_argument("file1", help="file to be read in", type=str)
parser.add_argument("file2", help="file to be read in", type=str)
args  = parser.parse_args() 

data_x=pd.read_csv(args.file1)
data_y=pd.read_csv(args.file2)

X=data_x
Y=data_y

# make training and test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=0)

#print(X_test) 
#print(y_test)

# fit total set to a lasso model 
alpha=0.1
print(alpha)
lasso=Lasso(alpha=alpha,normalize=True)
lasso.fit(X_train,y_train)
model_predict=lasso.predict(X_train)
print('####lasso coeff####')
print('Model weights:')
print(lasso.coef_)


print("Endpoint:")
print(np.transpose(np.array(y_train)))

print("Model:")
print(model_predict)
df1=pd.DataFrame((np.array(y_train)))
df2=pd.DataFrame(np.transpose(np.array(model_predict)))

df1.to_csv('lasso_model_vs_endpoint.csv')
df2.to_csv('lasso_model_vs_endpoint.csv',mode='a')


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


# Set the parameters by cross-validation
tuned_parameters = [{'normalize':[True],'alpha': [1e-4,1e-3,1e-2,1e-1,0,1e1,1e2,1e3]}]
scores = ['neg_mean_absolute_error']

for score in scores:
#    print(kernel[i])
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(Lasso(), tuned_parameters, cv=4, verbose=10,n_jobs=4, scoring='%s' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print('Score:')
    print(-clf.best_score_)
    print()

    #build models     
    model_test=clf.predict(X_test)
    model_train=clf.predict(X_train)
    r2_score_train=r2_score(y_train,model_train)
    mse_score_train=mean_squared_error(y_train,model_train)
    rmse_score_train=np.sqrt(mse_score_train)
    r2_score_test=r2_score(y_test,model_test)
    mse_score_test=mean_squared_error(y_test,model_test)
    rmse_score_test=np.sqrt(mse_score_test)

    #print('dft test')
    df1=pd.DataFrame(y_train)
    df2=pd.DataFrame(np.array(y_train)/model_train)
    #print(y_test)
    #print('predicted test') 
    #print(np.transpose(model_test))
    #print('dft train')
    #print(y_train)
    #print('predicted train')
    #print(np.transpose(model_train))

    #df1.to_csv('lasso_out_'+score+'_optimized.csv')
    #df2.to_csv('lasso_out_'+score+'_optimized.csv',mode='a')

    #plot figures
    plt.figure() 
    #plt.title('ML Performed via MLP on Training Set')
    plt.text(1.85,0.3,'Train:\nR2:%.3f \nMSE:%.3f\nRMSE:%.3f\nTest:\nR2:%.3f\nMSE:%.3f\nRMSE:%.3f' %(r2_score_train,mse_score_train,rmse_score_train,r2_score_test,mse_score_test,rmse_score_test), style='italic', bbox={'facecolor':'red', 'alpha':0.5, 'pad':10},fontsize=10) 
    plt.plot(y_train,model_train,'gs')
    plt.plot(y_train,y_train,'k-')
    plt.plot(y_test,model_test,'ro')
    plt.plot(y_test,y_test,'k-')
    #plt.axis([0,1.5,0,1.5])
    plt.xlabel('True')
    plt.ylabel('Model')
    plt.savefig('lasso_train_test'+score+'_optimized.png')

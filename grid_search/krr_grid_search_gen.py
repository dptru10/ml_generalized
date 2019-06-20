#!/usr/bin/env python 
from __future__ import print_function

import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np  
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score 
from sklearn.kernel_ridge import KernelRidge
import pandas as pd 
import argparse 

parser= argparse.ArgumentParser()
parser.add_argument("file1", help="file to be read in", type=str)
args  = parser.parse_args() 

data=pd.read_csv(args.file1)
features=['cnp','centro','vor_vol','vor_neigh']
endpoint='energy'

X=data[features]
Y=data[endpoint]

# make training and test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=0)

#print(X_test) 
#print(y_test)


# Set the parameters by cross-validation
tuned_parameters = [{'kernel':['rbf'],'coef0':[1e-4,1e-3, 1e-2, 1e2, 1e3],'gamma': [1e-4,1e-3, 1e-2, 1e2, 1e3],'alpha': [1e-3,1e-2,1e-1,0,1e1,1e2,1e3]}]
scores = ['neg_mean_absolute_error']
model='KRR'
for score in scores:
#    print(kernel[i])
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = RandomizedSearchCV(KernelRidge(), tuned_parameters, cv=5, verbose=10,n_jobs=-1, scoring='%s' % score)
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

    print("Endpoint:")
    print(np.transpose(np.array(y_train)))
    
    print("Model:")
    print(model_train)


    df1=pd.DataFrame()
    df2=pd.DataFrame()
	
    #df1['model_test']=pd.Series(model_test)
    #df1['true_test'] =pd.Series(y_test)
    df1['model_train']=pd.Series(model_train)
    df2['true_train'] =pd.Series(y_train)
    df1.to_csv('krr_model.csv',mode='w')
    df2.to_csv('krr_model_endpoint.csv',mode='w')
    print('Train:\nR2:%.3f \nMSE:%.3f\nRMSE:%.3f\nTest:\nR2:%.3f\nMSE:%.3f\nRMSE:%.3f' %(r2_score_train,mse_score_train,rmse_score_train,r2_score_test,mse_score_test,rmse_score_test))

    #plot figures
    plt.figure() 
    plt.title('ML Performed via %s on Training Set'%model)
    plt.text(np.max(y_train)-2.2*np.std(y_train),np.min(y_train)+(np.std(y_train)),'Train:\nR2:%.3f \nMSE:%.3f\nRMSE:%.3f\nTest:\nR2:%.3f\nMSE:%.3f\nRMSE:%.3f' %(r2_score_train,mse_score_train,rmse_score_train,r2_score_test,mse_score_test,rmse_score_test), style='italic', bbox={'facecolor':'red', 'alpha':0.5, 'pad':10},fontsize=10) 
    plt.scatter(x=y_train,y=model_train,c='green',marker='o',label='train')
    plt.plot(x=y_train,y=y_train)
    plt.scatter(x=y_test,y=model_test,c='red',marker='o',label='test')
    plt.legend(loc='upper left')
    #plt.axis([0,1.5,0,1.5])
    plt.xlabel('True')
    plt.ylabel('Model')
    plt.savefig('krr_train_test_'+score+'_optimized.png')

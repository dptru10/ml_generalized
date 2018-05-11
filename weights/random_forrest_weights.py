#!/usr/bin/env python 
import matplotlib.pyplot as plt
import numpy as np  
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, r2_score 
from sklearn.model_selection import train_test_split
import pandas as pd 
import argparse 

parser= argparse.ArgumentParser()
parser.add_argument("file1", help="file to be read in", type=str)
args  = parser.parse_args() 

data_test=pd.read_csv(args.file1)

#place features here
features=['#d elec/qn','covalent radius', 'electronegativity','ionization potential', 'electron affinity']#,'number of d electrons']

#place endpoint here
ydat='E_barrier'
X=data_test[features]
Y=data_test[ydat]

# make training and test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=0)
tuned_parameters = [{'n_estimators':[1,2,3,4,5]}]
scores = ['neg_mean_absolute_error']

forest = ExtraTreesRegressor(n_estimators=5,random_state=1)
forest.fit(X,Y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

 
# Print the feature ranking
print("Feature ranking:")

ranked_features=[]
for f in range(X.shape[1]):
    ranked_features.append(features[indices[f]])
    print("%d. %s (%f)" % (f + 1, features[indices[f]], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]),ranked_features,fontsize=6.5)
plt.xlim([-1, X.shape[1]])
plt.savefig('random_forrest_feature_importance.png')


for score in scores:
    forest = GridSearchCV(ExtraTreesRegressor(),tuned_parameters,cv=5,n_jobs=4,scoring='%s' %score)
    
    forest.fit(X_train, y_train)
    print("Best parameters set found on development set:")
    print()
    print(forest.best_params_)
    print()
    print('Score:')
    print(-forest.best_score_)
    print()
    
    model=forest.fit(X_train,y_train) 
    model_train=model.predict(X_train)
    model_test=model.predict(X_test)
    r2_score_train=r2_score(y_train,model_train)
    mse_score_train=mean_squared_error(y_train,model_train)
    rmse_score_train=np.sqrt(mse_score_train)
    r2_score_test=r2_score(y_test,model_test)
    mse_score_test=mean_squared_error(y_test,model_test)
    rmse_score_test=np.sqrt(mse_score_test)
    
    #plot figures
    plt.figure() 
    plt.title('ML Performed via RF on Training Set')
    plt.text(0.1, 1.2,'R2:%.3f \nMSE:%.3f\nRMSE:%.3f' %(r2_score_train,mse_score_train,rmse_score_train), style='italic', bbox={'facecolor':'red', 'alpha':0.5, 'pad':10},fontsize=10) 
    plt.plot(y_train,model_train,'gs')
    plt.plot(y_train,y_train,'k-')
    #plt.axis([0,1.5,0,1.5])
    plt.xlabel('True')
    plt.ylabel('Model')
    plt.savefig('rf_train_'+score+'_optimized.png')
    
    plt.figure() 
    plt.title('ML Performed via RF on Test Set')
    plt.text(0.85, 1.2, 'R2:%.3f\nMSE:%.3f\nRMSE:%.3f' %(r2_score_test,mse_score_test,rmse_score_test), style='italic', bbox={'facecolor':'red', 'alpha':0.5, 'pad':10},fontsize=10) 
    plt.plot(y_test,model_test,'gs')
    plt.plot(y_test,y_test,'k-')
    #plt.axis([0,2.5,0,2.5])
    plt.xlabel('True')
    plt.ylabel('Model')
    plt.savefig('rf_test_'+score+'_optimized.png')


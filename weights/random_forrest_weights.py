#!/usr/bin/env python 
from __future__ import print_function

import matplotlib 
matplotlib.use('Agg')
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

#change number of estimators
n_estimators=5

#place features here
features=['#d elec/qn','covalent radius', 'electronegativity','ionization potential', 'electron affinity']#,'number of d electrons']

#place endpoint here
ydat='E_barrier'
X=data_test[features]
Y=data_test[ydat]

# make training and test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=0)

forest = ExtraTreesRegressor(n_estimators=n_estimators,random_state=1)
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


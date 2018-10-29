#!/usr/bin/env python 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd 
import argparse

parser= argparse.ArgumentParser()
parser.add_argument("file1", help="file to be read in", type=str)
args  = parser.parse_args() 

data=pd.read_csv(args.file1)
features=['feature1','feature2','feature3', 'feature4','feature5']  
corr_dat = data[features]

# determine cross correlation table matrix
size=12
corr=corr_dat.corr()
fig, ax = plt.subplots(figsize=(size, size))
sns.set(font_scale=1.5)
sns_plot=sns.heatmap(corr,cmap="coolwarm",square=True,annot=True, fmt=".1f",annot_kws={"size": 20})
plt.yticks(rotation=45)
plt.xticks(rotation=45)
plt.savefig('cross_corr.png')

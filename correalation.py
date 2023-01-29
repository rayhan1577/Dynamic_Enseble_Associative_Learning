#!/usr/bin/env python3

import os
import sys
import sys
import time
from collections import defaultdict

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, confusion_matrix
import pandas as pd

import sigdirect
import dataset_transformation

class _Preprocess:

    def __init__(self):
        self._label_encoder = None

    def preprocess_data(self, raw_data):
        """ Given one of UCI files specific to SigDirect paper data,
            transform it to the common form used in sklearn datasets"""
        transaction_data =  [list(map(int, x.strip().split())) for x in raw_data]
        max_val = max([max(x[:-1]) for x in transaction_data])
        X,y = [], []

        for transaction in transaction_data:
            positions = np.array(transaction[:-1]) - 1
            transaction_np = np.zeros((max_val))
            transaction_np[positions] = 1
            X.append(transaction_np)
            y.append(transaction[-1])
        X = np.array(X)
        y = np.array(y)

        # converting labels
        if self._label_encoder is None:# train time
            unique_classes = np.unique(y)
            self._label_encoder = defaultdict(lambda: 0, zip(unique_classes, range(len(unique_classes))))
            y = np.vectorize(self._label_encoder.get)(y)
        else:# test time
            y = np.vectorize(lambda a: self._label_encoder[a])(y)

        return X,y

def prep_data(dataset_name,counter):
    import pandas as pd
    import warnings
    import pandas as pd
    import random
    from random import randint
    from sklearn.model_selection import train_test_split
    warnings.filterwarnings('ignore')
    
    input_name  = 'uci/' + dataset_name + '.txt' #train file
    
    #nameFile="datasets_new_originalfiles" +"\\"+fileName+".names"
    
    #method 1 for readin using pandas dataframe
    sep = ' '
    
    with open(input_name, 'r') as f:
        data = f.read().strip().split('\n')
    dataset = [line.strip().split(sep) for line in data]        
    
   
    df=pd.DataFrame(dataset)
    #print("df:",df )
    dforiginal=df
    #masking all the nones if any
    mask = df.applymap(lambda x: x is None)
    cols = df.columns[(mask).any()]
    for col in df[cols]:
        df.loc[mask[col], col] = ''
    dforiginal=df    
     
    dfnew=df
    #X = df.iloc[:,:-1].values
    #this line replaces all empty spaces with nan. this is done to get the last col values.
    dfnew = dfnew.apply(lambda x: x.str.strip() if isinstance(x, str) else x).replace('', np.nan)
    
    #now u can get the last col values that is the labels.
    Y=dfnew.groupby(['label'] * dfnew.shape[1], 1).agg('last')
    
    dfnew=dfnew.where(pd.notnull(dfnew), None)
    #dfnew = dfnew.apply(lambda x: x.str.strip() if isinstance(x, str) else x).replace(np.nan,None)
    dfnew=dfnew.values.tolist()
    
    print()
    for k in range(len(dfnew)):
        dfnew[k]=[x for x in dfnew[k] if x is not None]        #removing none from list
    
   
    
    print()
    for i in dfnew:
        i.pop()
    X=dfnew    
    
   
    #remove none and then then pop out the last element
    #masking all the nones if any
    
    
    #Y = df.iloc[:,-1].values # Here first : means fetch all rows :-1 means except last column
    #X = df.iloc[:,:-1].values # : is fetch all rows 3 means 3rd column  
    
    #print("len(X)",len(X))
    #print("len(Y)",len(Y))
    
    #X=np.array([np.array(xi) for xi in X])
    #X=np.asarray(X) 
    #print("Now x: ",X)
    
    X=pd.DataFrame(X)
    
    mask = X.applymap(lambda x: x is None)
    cols = X.columns[(mask).any()]
    for col in X[cols]:
        X.loc[mask[col], col] = '' 
   
    
    #print("X: ",X)
    #print("Y: ",Y)
    #print("Len of X: ",X.shape)
    #print("Len of Y: ",Y.shape)
  
    #now split the prune set
    #use smote
    # did smote here, then discretization on weka, but you can change that and do it inoython as well. and then tokenize it via data transformation code and then
    #use any sig classification model.
    '''
    print(len(Y))
    from imblearn.over_sampling import SMOTE
    smt = SMOTE(k_neighbors=5, random_state=1206)
    X, Y = smt.fit_sample(X, Y)
    print(len(Y))    
    dataset_transformation
    print(stopp)
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.10, shuffle='true')
    
    
         
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, shuffle='true')
    #X_val=X_train
    #y_val=y_train
    #print(X_train)
    #print(y_train)
    '''
    print("###########################len(X_train): ",len(X_train))
    
    print("len(X_test): ",len(X_test))
    print("len(y_train): ",len(y_train))
    print("len(y_test): ",len(y_test))
    print("len(X_val): ",len(X_val))
    print("len(y_val): ",len(y_val))
    '''
    X_test_original=X_test
    #df3_test=pd.DataFrame(list(X_test_original))              
    df_Xtest=X_test_original              
    #test_name  = 'datasets_new_ensemble_boosting/' + dataset_name + '_testoriginal'+ str(num_run) +'.txt'
    #df_Xtest.to_csv(test_name,sep=' ',index=False,header=False)  
    
    df3_test=y_test 
    #test_name  = 'datasets_new_ensemble_boosting/' + dataset_name + '_ytestoriginal'+ str(num_run) +'.txt'
    #df3_test.to_csv(test_name,sep=' ',index=False,header=False)          
            
    df_Xtrain=X_train                     
    #test_name  = 'datasets_new_ensemble_boosting/' + dataset_name + '_trainoriginal'+ str(num_run) +'.txt'
    #df_Xtrain.to_csv(test_name,sep=' ',index=False,header=False)
    
    df3_test=y_train                     
    #test_name  = 'datasets_new_ensemble_boosting/' + dataset_name + '_ytrainoriginal'+ str(num_run) +'.txt'
    #df3_test.to_csv(test_name,sep=' ',index=False,header=False)
    
    #you should make new datasets.
    #df1=pd.DataFrame(list(X_train))
    #df2=pd.DataFrame(list(y_train)) 
    #print(X_train)
    #print(y_train)
    df1=pd.DataFrame(X_train)
    df2=pd.DataFrame(y_train)   
    
    #for val set
    df1_val=pd.DataFrame(X_val)
    df2_val=pd.DataFrame(y_val)
    Xval_subsample = pd.concat([df1_val,df2_val], axis=1)#this is xval
    Xval_subsample=Xval_subsample.reset_index(drop=True)
    Xval_subsample.columns = list(range(0, X_train.shape[1]+1))        
    #print("Xval_subsample: ",Xval_subsample)       
    #print("NOW??????????????????df1: ",df1)
    
    
    df4y_test=pd.DataFrame(X_test) 
    y_test=pd.DataFrame(y_test)
    Xtest_subsample = pd.concat([df4y_test,y_test], axis=1) #this is xtest
    Xtest_subsample=Xtest_subsample.reset_index(drop=True)
    Xtest_subsample.columns = list(range(0, X_train.shape[1]+1))
    
    Xtotal_subsample = pd.concat([df1,df2], axis=1) #this is xtotal
    
    Xtotal_subsample=Xtotal_subsample.reset_index(drop=True)
    Xtotal_subsample.columns = list(range(0, X_train.shape[1]+1))
    
    test_name  = 'datasets_new_ensemble_boosting/' + dataset_name + '_validation'+ str(counter) +'.txt'    
    Xval_subsample.to_csv(test_name,sep=' ',index=False,header=False)  
    
    test_name  = 'datasets_new_ensemble_boosting/' + dataset_name + '_train'+ str(counter) +'.txt'
    Xtotal_subsample.to_csv(test_name,sep=' ',index=False,header=False)
    
    test_name  = 'datasets_new_ensemble_boosting/' + dataset_name + '_test'+ str(counter) +'.txt'
    Xtest_subsample.to_csv(test_name,sep=' ',index=False,header=False)    
import seaborn as sns

from sklearn.model_selection import train_test_split
def test_uci():
    dataset_name="anneal"
    prep = _Preprocess()
    train_filename = os.path.join('uci', '{}.txt'.format(dataset_name))
    with open(train_filename) as f:
        raw_data = f.read().strip().split('\n')
    X, y = prep.preprocess_data(raw_data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_train=pd.DataFrame(y_train)
    #X_train=pd.DataFrame(X_train)
    df=X_train.copy()
    df=pd.DataFrame(df)
    X_train=pd.DataFrame(X_train)
    X_test=pd.DataFrame(X_test)
    df[df.shape]=y_train
    correlation_df =df.corr()

    sns.heatmap(df.corr())
    

if __name__ == '__main__':
    test_uci()

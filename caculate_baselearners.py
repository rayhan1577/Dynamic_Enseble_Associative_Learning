
"""
This is modified version of version 4. here we changed the sampling. Choose 25 sample at first and then took 12 old features and 13 new features each time

"""
import os
import sys
import time
from collections import defaultdict
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
import sigdirect
from random import seed
from random import random
from random import randrange


class _Preprocess:

    def __init__(self):
        self._label_encoder = None

    def preprocess_data(self, raw_data):
        """ Given one of UCI files specific to SigDirect paper data,
            transform it to the common form used in sklearn datasets"""
        transaction_data = [list(map(int, x.strip().split())) for x in raw_data]
        max_val = max([max(x[:-1]) for x in transaction_data])
        X, y = [], []

        for transaction in transaction_data:
            positions = np.array(transaction[:-1]) - 1
            transaction_np = np.zeros((max_val))
            transaction_np[positions] = 1
            X.append(transaction_np)
            y.append(transaction[-1])
        X = np.array(X)
        y = np.array(y)

        # converting labels
        if self._label_encoder is None:  # train time
            unique_classes = np.unique(y)
            self._label_encoder = defaultdict(lambda: 0, zip(unique_classes, range(len(unique_classes))))
            y = np.vectorize(self._label_encoder.get)(y)
        else:  # test time
            y = np.vectorize(lambda a: self._label_encoder[a])(y)

        return X, y


def test_uci():
    #"wine","flare",
    data=["pima","glass","pageBlocks","heart","hepati","wine","anneal","horse","mushroom","adult","ionosphere"]
    #data=["adult","ionosphere","penDigits","mushroom","soybean","cylBands"]
    #data=["iris"]
    for dataset_name in data:
        memory=0
        start_time = time.time()
        #print(ratio, end=" ")
        print(dataset_name, end=" ")
        all_pred_y = defaultdict(list)
        all_true_y = []

        # counting number of rules before and after pruning
        generated_counter = 0
        final_counter = 0
        avg = [0.0] * 4

        tt1 = time.time()
        predictors = []
        # print(index)
        prep = _Preprocess()

        # load the training data and pre-process it
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
        correlation_df=correlation_df.to_numpy()
        #print(correlation_df[0])
        correlation_df[-1]=np.nan_to_num(correlation_df[-1])
        #print(correlation_df[0])
        correlations=np.copy(correlation_df[-1])
        correlations.sort()

        correlations=correlations[:len(correlations)-1]
        #all_correlations=correlations[0]
        list_features=[]
       
        #print(correlation_df[0])

        for i in range(len(correlations)):
            for j in range(len(correlation_df[-1])):
                if(round(correlations[i],4)==round(correlation_df[-1][j],4)):
                    if j not in list_features:
                        list_features.append(j)
       
        #least=list_features[:int(len(list_features)/2)]
        #most=list_features[int(len(list_features)/2):len(list_features)-1]
        all_subsample = subsample(list_features)
        
# Create a random subsample from the dataset with replacement
import random
def unique(list1):
    x = np.array(list1)
    return np.unique(x)

def subsample(list_features):
    features=[]
    feature_list=[]

    taken_features=[]
    all_subsample=[]
    feat=25
    ov=15
    tracker=0
    f=list_features.copy()
    #print(len(list_features))
    subsampling=True
    while(subsampling and len(unique(taken_features))!=len(f)):
        new_subsample=[]
        if(len(f)<feat):
            all_subsample.append(f)
            break
        else:
            for i in range(feat):
                random.seed()
                index = random.randint(0, len(list_features))
                new_subsample.append(index)

            all_count=[]           
            for c in all_subsample:
                count=0           
                for i in new_subsample:
                    if(i in c):
                        count+=1
                #print("count, n_sample,int(overlap)")
                all_count.append(count)
            #print(all_count)   
            #input()
            c2 = [i for i in all_count if i >ov]

            if(len(c2)>0):
                tracker+=1
                if(tracker>=100):
                    break
            else:
                #print("hello")
                all_subsample.append(new_subsample)
                taken_features.extend(new_subsample)
                tracker=0


    print(len(all_subsample))


       
        




if __name__ == '__main__':
    test_uci()
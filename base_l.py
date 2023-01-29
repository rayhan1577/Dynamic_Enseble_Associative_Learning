
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
    print(count_base_learner(35))
        
# Create a random subsample from the dataset with replacement
import random


def count_base_learner(l):
    list_features=[]
    for i in range(l):
        list_features.append(i)    
    features=[]
    taken_features=[]
    all_subsample=[]
    feat=25
    ov=15
    tracker=0
    f=list_features.copy()
    #print(len(list_features))
    subsampling=True
    while(subsampling):
        new_subsample=[]
        for i in range(feat):
            random.seed()
            index = random.randint(0, len(list_features))
            new_subsample.append(index)
        #print(len(new_subsample))
        all_count=[]           
        for c in all_subsample:
            x=list(set(c).intersection(new_subsample))
            print(x)
            all_count.append(len(x))
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
            tracker=0


    return len(all_subsample)



       
        




if __name__ == '__main__':
    test_uci()
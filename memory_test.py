import os
import sys
import time
from collections import defaultdict

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import sigdirect
import pandas as pd

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


    dataset=["iris","breast","glass","heart","hepati","wine","pima","zoo","flare","led7","pageBlocks","anneal","horse"]
    #dataset=["iris"]
    for dataset_name in dataset:
        print(dataset_name, end=" ")
        # counting number of rules before and after pruning
        generated_counter = 0
        final_counter = 0
        avg = [0.0] * 4

        tt1 = time.time()

        prep = _Preprocess()

        # load the training data and pre-process it
        train_filename = os.path.join('uci', '{}.txt'.format(dataset_name))
        with open(train_filename) as f:
            raw_data = f.read().strip().split('\n')
        X, y = prep.preprocess_data(raw_data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        if(X_train.shape[1]<25):
            clf = sigdirect.SigDirect(get_logs=sys.stdout)
            generated_counter, final_counter,memory = clf.fit(X_train, y_train)



            # evaluate the classifier using different heuristics for pruning
            for hrs in (1, 2, 3):
                y_pred = clf.predict(X_test, hrs)
                #print('ACC S{}:'.format(hrs), accuracy_score(y, y_pred))
                #print(accuracy_score(y_test, y_pred), end=" ")
            

            #print(generated_counter , final_counter, end=" ")
            print( memory, memory)
        else:
            clf = sigdirect.SigDirect(get_logs=sys.stdout)
            generated_counter, final_counter,memory1 = clf.fit(X_train, y_train)
            temp=[]
            df = pd.DataFrame(X_train)
            df2 = pd.DataFrame(X_test)
            sample = pd.DataFrame()
            test_sample = pd.DataFrame()
            for i in range(25):
                index=random.randint(0,X_train.shape[1]-1)
                if (index not in temp):
                    sample[sample.shape] = df[index]
                    test_sample[test_sample.shape] = df2[index]
                    temp.append(index)
            
            sample = np.array(sample)
            test_sample = np.array(test_sample)
            clf = sigdirect.SigDirect(get_logs=sys.stdout)
            g, f,memory2 = clf.fit(sample, y_train)
            for hrs in (1, 2, 3):
                y_pred = clf.predict(test_sample, hrs)
                #print('ACC S{}:'.format(hrs), accuracy_score(y, y_pred))
                #print(accuracy_score(y_test, y_pred), end=" ")
            

            #print(generated_counter , final_counter, end=" ")
            print( memory1, memory2)
                    
            

import random
if __name__ == '__main__':
    #start_time = time.time()
    test_uci()
    #end_time=time.time()
    #print("required_time:", end_time-start_time)
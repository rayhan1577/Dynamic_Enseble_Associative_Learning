import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,precision_score,recall_score,f1_score
import matplotlib.pyplot as plt 

import time, psutil

import os
import sys
import time
from collections import defaultdict

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

t1=time.time()

prep = _Preprocess()
with open('adult.txt') as f:
    raw_data = f.read().strip().split('\n')
X, y = prep.preprocess_data(raw_data)







X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
#print('X_train :',X_train.shape)
#print('y_train :',y_train.shape)
#print('X_test :',X_test.shape)
#print('y_test :',y_test.shape)



from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)






#print("accuracy",accuracy_score(y_test, y_pred), "precission",precision_score(y_test, y_pred,average='macro'),"recall",recall_score(y_test, y_pred,average='macro'), end=" ")
#print(y_pred.shape)
print("accuracy",accuracy_score(y_test, y_pred))

#WHOLE PROGRAM
process = psutil.Process(os.getpid())
print("memory",process.memory_info().rss/ 1024 ** 2)
t2=time.time()
print("time",t2-t1)
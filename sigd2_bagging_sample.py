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
    assert len(sys.argv) > 1
    dataset_name = sys.argv[1]


    if len(sys.argv) > 2:
        start_index = int(sys.argv[2])
    else:
        start_index = 1

    final_index = 10
    k = final_index - start_index + 1
    #dataset=["Iris","Breast","Heart","Wine","Pima","Zoo","Flare","Led7","glass","Anneal","Hepati","Horse","Adult","pageblocks"]
    dataset=["Anneal","Hepati","Horse","Adult","pageblocks","flare"]
    for dataset_name in dataset:
        print(dataset_name, end=" " )
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
        generated_c = 0
        final_c = 0
        """
        clf = sigdirect.SigDirect(get_logs=sys.stdout)
        g, f = clf.fit(X_train, y_train)
        for i in (1, 2, 3):
            y_pred=clf.predict(X_test, i)
            print('ACC S{}:'.format(i), accuracy_score(y_test, y_pred))
        """
        pred1 = []
        pred2 = []
        pred3 = []
        for i in range(50):
            clf = sigdirect.SigDirect(get_logs=sys.stdout)
            seed(1)
            train_filename = os.path.join('uci', '{}.txt'.format(dataset_name))
            with open(train_filename) as f:
                raw_data = f.read().strip().split('\n')

            X, y =subsample(X_train, y_train)
            X = np.array(X)
            y = np.array(y)
            g, f = clf.fit(X, y)
            pred1.append(clf.predict(X_test, 1))
            #print(pred1)
            pred2.append(clf.predict(X_test, 2))
            pred3.append(clf.predict(X_test, 3))

            generated_c += g
            final_c += f

        generated_counter = generated_c / 50
        final_counter = final_c / 50

        final_prediction = []
        pred1 = np.array(pred1)
        pred1 = pred1.transpose()
        for i in pred1:
            i = list(i)
            final_prediction.append(max(i, key=i.count))
        #print('ACC S{}:'.format("1"), accuracy_score(y_test, final_prediction))
        print("{:.4f}".format(accuracy_score(y_test, final_prediction)),end=" ")
        final_prediction = []
        pred2 = np.array(pred2)
        pred2 = pred2.transpose()
        for i in pred2:
            i = list(i)
            final_prediction.append(max(i, key=i.count))
        #print('ACC S{}:'.format("2"), accuracy_score(y_test, final_prediction))
        print("{:.4f}".format(accuracy_score(y_test, final_prediction)),end=" ")
        final_prediction = []
        pred3 = np.array(pred3)
        pred3 = pred3.transpose()
        for i in pred3:
            i = list(i)
            final_prediction.append(max(i, key=i.count))
        #print('ACC S{}:'.format("3"), accuracy_score(y_test, final_prediction))
        #print('INITIAL RULES: {} ---- FINAL RULES: {}'.format(generated_counter, final_counter))
        end_time = time.time()
        print("{:.4f}".format(accuracy_score(y_test, final_prediction)), end=" ")
        print(generated_counter, final_counter, "{:.2f}".format(time.time() - start_time) )
        #print("required time:", end_time - start_time)



# Create a random subsample from the dataset with replacement
import random


def subsample(dataset, y):
    sample = list()
    y_sample = list()
    n_sample = round(len(dataset) * .6)
    temp=[]
    seed()
    while len(sample) < n_sample:
        index = random.randrange(len(dataset))
        if index not in temp:
            temp.append(index)
            sample.append(dataset[index])
            y_sample.append(y[index])
    return sample, y_sample


def mean(numbers):
    return sum(numbers) / float(len(numbers))


if __name__ == '__main__':
    start_time=time.time()
    test_uci()
    end_time=time.time()
    #print("required time", end_time-start_time)
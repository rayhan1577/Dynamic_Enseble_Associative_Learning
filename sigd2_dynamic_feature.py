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
    print(dataset_name)

    if len(sys.argv) > 2:
        start_index = int(sys.argv[2])
    else:
        start_index = 1

    final_index = 10
    k = final_index - start_index + 1

    dataset=[ "glass"]
    
    for dataset_name in dataset:

        #print(dataset_name, end=" ")

        predictors = []
        # print(index)
        collect.clear()
        all_pred_y = defaultdict(list)
        all_true_y = []

        # counting number of rules before and after pruning
        generated_counter = 0
        final_counter = 0
        avg = [0.0] * 4

        tt1 = time.time()
        for ratio in [.7]:
            prep = _Preprocess()
            print(dataset_name, ratio, end=" ")
            # load the training data and pre-process it
            train_filename = os.path.join('uci', '{}.txt'.format(dataset_name))
            with open(train_filename) as f:
                raw_data = f.read().strip().split('\n')

            X, y = prep.preprocess_data(raw_data)
            X=X[:300,:]
            y=y[:300]
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
            no_of_predictor=0
            for i in range(10):
                print(i)
                clf = sigdirect.SigDirect(get_logs=sys.stdout)
                seed(1)
                # True mean
                dataset = [[randrange(len(X_train))] for i in range(20)]

                sample, sample_test = subsample(X_train,  X_test, ratio)
                if(not isinstance(sample, pd.core.frame.DataFrame)):
                        break
                #sample,sample_test=feature_select(sample, X_test)
                no_of_predictor+=1

                sample = np.array(sample)
                sample_test = np.array(sample_test)
                # print("sample size",len(sample))
               
                g, f ,m= clf.fit(sample, y_train)
                pred1.append(clf.predict(sample_test, 1))
                #print('ACC of one run S{}:'.format("1"), accuracy_score(y_test, clf.predict(X_test, 1)))
                pred2.append(clf.predict(sample_test, 2))
                pred3.append(clf.predict(sample_test, 3))

                generated_c += g
                final_c += f

            generated_counter = generated_c / no_of_predictor
            final_counter = final_c / no_of_predictor

            final_prediction = []
            pred1 = np.array(pred1)
            pred1 = pred1.transpose()
            for i in pred1:
                i = list(i)
                final_prediction.append(max(i, key=i.count))
            #print('ACC S{}:'.format("1"), accuracy_score(y_test, final_prediction))
            print(str(round(accuracy_score(y_test, final_prediction),4)), end=" ")

            final_prediction = []
            pred2 = np.array(pred2)
            pred2 = pred2.transpose()
            for i in pred2:
                i = list(i)
                final_prediction.append(max(i, key=i.count))
            #print('ACC S{}:'.format("2"), accuracy_score(y_test, final_prediction))
            print(str(round(accuracy_score(y_test, final_prediction),4)), end=" ")

            final_prediction = []
            pred3 = np.array(pred3)
            pred3 = pred3.transpose()
            for i in pred3:
                i = list(i)
                final_prediction.append(max(i, key=i.count))
            #print('ACC S{}:'.format("3"), accuracy_score(y_test, final_prediction))
            print(str(round(accuracy_score(y_test, final_prediction),4)), end=" ")
            #print('INITIAL RULES: {} ---- FINAL RULES: {}'.format(generated_counter, final_counter))
            print(generated_counter, final_counter, end=" ")
            end_time = time.time()
            #print("required time:", end_time - start_time)
            x=end_time - start_time
            print(float("{0:.2f}".format(x)), no_of_predictor)



# Create a random subsample from the dataset with replacement
import random
collect=[]

def subsample(dataset, test_dataset,ratio):
    df = pd.DataFrame(dataset)
    df2 = pd.DataFrame(test_dataset)
    run_loop=True
    random.seed()
    tracker=0 #count how many times subsampling done
    all_count=[]
    while(run_loop and tracker<10):
        sample = pd.DataFrame()
        test_sample = pd.DataFrame()
        y_sample = pd.DataFrame()
        index = []
        n_feature = df.shape[1]
        if (n_feature > 30):
            n_feature = 30
        else:
            n_feature = n_feature*.6
        temp = []
        count=0
        random.seed()
        while (len(temp) < n_feature):
            index = random.randint(0, df.shape[1] - 1)
            if (index not in temp):
                temp.append(index)
                # print(index)
                sample[sample.shape] = df[index]
                test_sample[test_sample.shape] = df2[index]
                
        if(len(collect)==0):
            break  
                   
        for c in collect:
            count=0           
            for i in temp:
                for j in c:
                    if(i==j):
                        count+=1
            #print(count, n_sample,int(overlap))
            all_count.append(count)
            
        c2 = [i for i in all_count if i > int(len(temp)*ratio)]    
        if(len(c2)>0):
            tracker+=1
        else:
            break

    if(tracker<10):
        collect.append(list(temp))
        return sample, test_sample
    else:    
        return -1,-1
    


def mean(numbers):
    return sum(numbers) / float(len(numbers))

import random
#count_diverse=[]
def feature_select(dataset, test_dataset):
    df = pd.DataFrame(dataset)
    df2 = pd.DataFrame(test_dataset)
    sample = pd.DataFrame()
    test_sample = pd.DataFrame()
    y_sample = pd.DataFrame()
    index = []
    n_feature = df.shape[1]
    n_feature = int(n_feature*.6)
    temp = []

    random.seed()
    while (len(temp) < n_feature):
        index = random.randint(0, df.shape[1] - 1)
        if (index not in temp ):#and count_diverse.count(index)<300/df.shape[1] + 1
            temp.append(index)
            # print(index)
            #count_diverse.append(index)
            sample[sample.shape] = df[index]
            test_sample[test_sample.shape] = df2[index]
    x = sample.values.tolist()
    return sample, test_sample


if __name__ == '__main__':
    start_time=time.time()
    test_uci()
    end_time=time.time()
    print("required time", end_time-start_time)

"""
In this version is clean version for version 2. Here we first made the subsamples and then trained. In the subsamples there is no dataset. Only sampling of the features.

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
            print(unique_classes)
            self._label_encoder = defaultdict(lambda: 0, zip(unique_classes, range(len(unique_classes))))
            y = np.vectorize(self._label_encoder.get)(y)
        else:  # test time
            y = np.vectorize(lambda a: self._label_encoder[a])(y)

        return X, y


def test_uci():

    #data=["iris","breast","glass","heart","hepati","wine","pima","zoo","flare","led7","pageBlocks","anneal","horse","adult","ionosphere","penDigits","mushroom","soybean","cylBands"]
    data=["soybean"]
    #data=["iris"]
   
    for dataset_name in data:
        memory=0
        start_time = time.time()
        #print(ratio, end=" ")
        print(dataset_name, )
        
        prep = _Preprocess()

        # load the training data and pre-process it
        train_filename = os.path.join('uci', '{}.txt'.format(dataset_name))
        with open(train_filename) as f:
            raw_data = f.read().strip().split('\n')

        X, y = prep.preprocess_data(raw_data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if(X.shape[1]<38):
            clf = sigdirect.SigDirect(get_logs=sys.stdout)
            generated_counter, final_counter,memory = clf.fit(X_train, y_train)
            # evaluate the classifier using different heuristics for pruning
            for hrs in (1, 2, 3):
                y_pred = clf.predict(X_test, hrs)
                #print('ACC S{}:'.format(hrs), accuracy_score(y, y_pred))
                print(accuracy_score(y_test, y_pred), end=" ")
            

            print(generated_counter , final_counter, end=" ")
            print( time.time() - start_time,1, memory)
        else:
            all_pred_y = defaultdict(list)
            all_true_y = []

            # counting number of rules before and after pruning
            generated_counter = 0
            final_counter = 0
            avg = [0.0] * 4

            tt1 = time.time()
            predictors = []
            # print(index)
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
            #for i in all_subsample:
            #    print(i)
            #input("hold")
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
            for sample in all_subsample:
                

                train_data=  pd.DataFrame()
                test_data =  pd.DataFrame()
                #print("x test lenght",len(X_test))
                for index in sample:
                    train_data[train_data.shape] = X_train[index]
                    test_data[test_data.shape] = X_test[index]
                train_data=np.array(train_data)
                test_data=np.array(test_data)
                #print("test shape",test_data.shape)
                clf = sigdirect.SigDirect(get_logs=sys.stdout)
                g, f,m = clf.fit(train_data, y_train)
                if(m>memory):
                    memory=m
                pred1.append(clf.predict(test_data, 1))
                #print(pred1)
                pred2.append(clf.predict(test_data, 2))
                pred3.append(clf.predict(test_data, 3))

                generated_c += g
                final_c += f
            no_of_predictor=len(all_subsample)
            generated_counter = generated_c / no_of_predictor
            final_counter = final_c / no_of_predictor

            final_prediction = []
            pred1 = np.array(pred1)
            pred1 = pred1.transpose()
            for i in pred1:
                i = list(i)
                final_prediction.append(max(i, key=i.count))
            #print(len(y_test),len(final_prediction))
            print( accuracy_score(y_test, final_prediction), end=" " )

            final_prediction = []
            pred2 = np.array(pred2)
            pred2 = pred2.transpose()
            for i in pred2:
                i = list(i)
                final_prediction.append(max(i, key=i.count))
            print( accuracy_score(y_test, final_prediction), end=" " )

            final_prediction = []
            pred3 = np.array(pred3)
            pred3 = pred3.transpose()
            for i in pred3:
                i = list(i)
                final_prediction.append(max(i, key=i.count))
            print( accuracy_score(y_test, final_prediction), end=" " )
            print(generated_counter, final_counter, end=" " )
            end_time = time.time()
            print( end_time - start_time, len(all_subsample), memory)
            """
            # load the test data and pre-process it.
            test_filename = os.path.join('uci', '{}_ts{}.txt'.format(dataset_name, index))
            with open(test_filename) as f:
                raw_data = f.read().strip().split('\n')
            X, y = prep.preprocess_data(raw_data)
            # evaluate the classifier using different heuristics for pruning
            for hrs in (1,2,3):
                y_pred = clf.predict(X, hrs)
                print('ACC S{}:'.format(hrs), accuracy_score(y, y_pred))
                avg[hrs] += accuracy_score(y, y_pred)
                all_pred_y[hrs].extend(y_pred)
            """
            # all_true_y.extend(list(y))
            # print('\n\n')

            """
            print("final score")
            print("=================")
            print("accuracy",accuracy_score(y, final_prediction))
            print(dataset_name)
            for hrs in (1,2,3):
                print('AVG ACC S{}:'.format(hrs), accuracy_score(all_true_y, all_pred_y[hrs]))
            print('INITIAL RULES: {} ---- FINAL RULES: {}'.format(generated_counter/k, final_counter/k))
            print('TOTAL TIME:', time.time()-tt1)
            """
			



# Create a random subsample from the dataset with replacement
import random


def subsample(list_features):
    all_subsample=[]
    feat=30   
    random.seed(time.time())
    tracker=0 #count how many times subsampling done
    while tracker<100:
        #print(len(list_features), len(all_subsample))
        l1=list_features.copy()
        new_subsample=[]
        if(len(list_features)<25):
           all_subsample.append(list_features)
           #print("test",all_subsample)
           return all_subsample
        elif(len(all_subsample)==0):
            for i in range(feat):
                new_subsample.append(list_features[i])
            all_subsample.append(new_subsample)
        else:
            for i in range(feat):
                x=random.choice(l1)
                l1.remove(x)
                new_subsample.append(x)
       
            all_count=[]           
            for c in all_subsample:
                count=0           
                for i in new_subsample:
                    if(i in c):
                        count+=1
                #print(count, n_sample,int(overlap))
                all_count.append(count)
            #print(all_count)   
            c2 = [i for i in all_count if i >18]

            if(len(c2)>0):
                tracker+=1
            else:
                all_subsample.append(new_subsample)
                tracker=0
    return all_subsample


       
        




if __name__ == '__main__':
    test_uci()
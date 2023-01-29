# SigDirect
This is a python implementation of SigDirect [1] classifier. 
The classifier does not have any specific hyper-parameter to tune and 
the only one is the p-value of the statistical significance, 
which is set to 0.05 (you can change it in the config file, though this is the value used in most scientific works)

## Running
You first need to use the requirements.txt file to install the dependencies:
```
pip3 install -r requirements.txt
```

You can test the dataset using one of the provided UCI datasets:
```
python3 sigdirect_test.py iris
```
In order to use the classifier, you can call the code similar to scikit-learn classifiers:
You should instantiate it first, and then call the ```fit``` method to train a model. 
To get predictions, you can either call ```predict``` or ```predict_proba``` methods where 
the former will provide classes and the latter will provide the probability distributions. 

The input to fit method should be a 2-d numpy array where each feature can be either 0 or 1. 
For labels, a 1-d array should be used where each element should also be integers (starting from 0 to n-1 for n classes.)

Note: You can provide ```get_logs=std.out``` or ```get_logs=std.err``` as an argument to the constructor
of the classifier so it will print some logs about creating the model. 

[1]: Li, Jundong, and Osmar R. Zaiane. "Exploiting statistically significant dependent rules for associative classification." Intelligent Data Analysis 21.5 (2017): 1155-1172.
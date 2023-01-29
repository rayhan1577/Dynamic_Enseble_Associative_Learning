import sklearn
from sklearn import preprocessing
import numpy as np
import pandas as pd

import csv
#nameFile="D:/Thesis/summer_2019/sigdirect_float_Try - Copy\datasets_new_originalfiles\WCB.txt"
#you can use this code after discretizing data from weka, this will change it to sig format in the continuous numerical format. - 2 April 2020


def transformdata(X,df_y_train):
            
    
    #dataset=X
    print("X: ",X)
    #print("Xval: ",Xval)
    #print(weigh)
    #df=pd.DataFrame(dataset)
    df=X
    df=df.reset_index(drop=True)
    print(df )
    
    
    dforiginal=df
    #masking all the nones if any
    mask = df.applymap(lambda x: x is None)
    cols = df.columns[(mask).any()]
    for col in df[cols]:
        df.loc[mask[col], col] = ''
    dforiginal=df    
    dfnew=df
    #print("dfnew")
    #print(dfnew)
    
    #print("X: ",X)
    
    import sklearn
    from sklearn import preprocessing
    import numpy as np
    enc = sklearn.preprocessing.OrdinalEncoder()
    
    X=pd.DataFrame(X)
    
    #enc.fit(X)
    #X_transformed=enc.transform(X)
    
    #X_transformed=pd.DataFrame(X_transformed)
    X_transformed=X
    #for val
    #enc.fit(Xval)
    #X_transformed_val=enc.transform(Xval)
    #print("Xval AFTER")
    #X_transformed_val=pd.DataFrame(X_transformed_val)    
    #X_transformed_val=Xval
    
    temp=[]
    temp=list(range(0,X_transformed.shape[1]))
    #df = pd.DataFrame(np.random.randn(8, 4),columns=temp)  
    X_transformed.columns=temp
    #X_transformed_val.columns=temp
    #print("X_transformed: ",X_transformed)
    #print("X_transformed_val: ",X_transformed_val)
    
    #print("type of X_transformed:", type(X_transformed))
    X_transformed_perrange = X_transformed
    #X_transformed_perrange_val = X_transformed_val
    X_transformed_perrange=X_transformed_perrange.astype(str)
    #X_transformed_perrange_val=X_transformed_perrange_val.astype(str)
    
    #print("X_transformed_perrange: ",X_transformed_perrange)
    #print("X_transformed_perrange_val: ",X_transformed_perrange_val)
    #print("focusX_transformed.shape[1]: ",X_transformed.shape[1])
    #input_name='datatransform/' + 'wine_train0'+'.txt' #csv
    #set_val=set()
    #set_train=set()
    count=1
    tempcount=0
    for i in range(0,X_transformed.shape[1]):        
        #print("??X_transformed[i]: ",X_transformed[i])
        
        #print("i: ",i)
        #print("type of X_transformed[i]:", type(X_transformed[i]))
        #print("set of X_transformed[i] : ",set(X_transformed[i]))
        #print("unique method: ",X_transformed[i].unique())
        #set_train=X_transformed[i].unique()
        #set_val=X_transformed_val[i].unique()
        set_train=list(X_transformed[i].unique())
        #set_val=list(X_transformed_val[i].unique())        
        #superset=set_train.union(set_val)
        #set_train=list(map(int, set_train))
        #set_val=list(map(int, set_val))
        superset=list(set(set_train))
        #print("set_train: ",set_train)
        #print("set_val: ",set_val)
        #print("superset: ",superset)
        for element in superset:        
            #print("X_transformed_perrange[i]: before ",X_transformed_perrange[i])
            #print("X_transformed_perrange_val[i]: before ",X_transformed_perrange_val[i])
            #print("looking for element: replacing it with value:  ",str(element),str(count))
            #if X_transformed_perrange.isin
            #X_transformed_perrange[i].replace([int(element)], [int(count)],inplace=True)
            #X_transformed_perrange_val[i].replace([int(element)], [int(count)],inplace=True)
            #print("??????????: ",X_transformed_perrange[i].isin([str(element)]))
            X_transformed_perrange[i]=X_transformed_perrange[i].replace(str(element), str(count))
            #X_transformed_perrange_val[i]=X_transformed_perrange_val[i].replace(str(element), str(count))           
            
            #print("??????????: ",X_transformed_perrange[i].isin([int(element)]))
            #print("X_transformed_perrange[i]: after ",X_transformed_perrange[i])
            #print("X_transformed_perrange_val[i]: after ",X_transformed_perrange_val[i])            
            count+=1
        
        #if i==2:
        #    print(stopit)
            #
        #count=max(superset) #some problem with count solve it 
        #print("count: ",count)
        
        #print(eighh)
        #if set difference is zero basically take union
        # sort them in asc order. then for each element in unioned set.
        #find that value in both the df's. and map it with the new values
        #what will be the new value?
        
        #if set differ is not zero find the bigger one. union type
        
        
       #print(weigh)
       #print("X_transformed[i].nunique()",X_transformed[i].nunique())
       
        temp.append(count)
    #print("X_transformed_perrange: ",X_transformed_perrange)
    #print("X_transformed_perrange_val: ",X_transformed_perrange_val)
    #print(stopcheckcheckk)
    temp=[]    
    X_transformed_perrange=X_transformed_perrange.astype(float)
    X_transformed_perrange=X_transformed_perrange.astype(int)
    #X_transformed_perrange_val=X_transformed_perrange_val.astype(int)
    #X_transformed_perrange.to_csv(input_name,sep=' ',index=False,header=False)
    #print("X_transformed_perrange")
    #print(X_transformed_perrange)
    #print("last col index: ",i)    
    count1=X_transformed_perrange.iloc[:,-1].max()
    #count2=X_transformed_perrange_val.iloc[:,-1].max()
    #count=int(max(int(count1)))
    count=int(count1)
    counter=int(count)+1
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@2X transformed: ",X_transformed_perrange)
    print("count1??????: ",count1)
    enc = sklearn.preprocessing.OrdinalEncoder()
    print("df_y_train: ",df_y_train)
    df_y_train=pd.DataFrame(df_y_train)
    enc.fit(df_y_train)
    df_y_train=enc.transform(df_y_train)    
    df_y_train=pd.DataFrame(df_y_train)  
    #enc.fit(Yval)
    #Yval=enc.transform(Yval)    
    #Yval=pd.DataFrame(Yval)    
    
    #print("df_y_train: ",df_y_train)
    #print("type of df_y_train: ",type(df_y_train))
    #print("counter: ",counter)
    #print("type of counter: ",type(counter))
    df_y_train=df_y_train+int(counter) #['label']
    #Yval=Yval+int(counter) #['label']
    #print("final X_transformed_perrange: ",X_transformed_perrange) 
    #print("df_y_train: ",df_y_train)
    X_transformed_perrange=X_transformed_perrange.astype(float)
    X_transformed_perrange=X_transformed_perrange.astype(int)
    df_y_train=df_y_train.astype(int)
    #X_transformed_perrange_val=X_transformed_perrange_val.astype(int)
    #Yval=Yval.astype(int)  
    print("X_transformed_perrange: ",X_transformed_perrange)
    print("df_y_train: ",df_y_train)
    #print("X_transformed_perrange_val: ",X_transformed_perrange_val)
    #print("Yval: ",Yval)
    
    #print(weight)
    #print(stop)
    return X_transformed_perrange,df_y_train
import numpy as np



def main():
    import csv
    #nameFile="D:/Thesis/summer_2019/sigdirect_float_Try - Copy\datasets_new_originalfiles\WCB.txt"
    
    nameFile= './data_covid_smoted_DISCRETIZED'+'.csv' #csv
    
    fileContent=open(nameFile,"r")
    fileCont=fileContent.readlines()
    dataset=[]
    
    for row in fileCont:
        row=row.strip()
        #row=row.replace("?"," ")
        dataset.append(row.split(','))
    
    print("dataset")    
    dataset=pd.DataFrame(dataset)
    print(dataset)
    X=dataset.iloc[:,:-1]
    Y=dataset.iloc[:,-1]
    print(X)
    print(Y)
    
    
    X_transformed_perrange,df_y_train =transformdata(X,Y)
    Xtotal=pd.concat([X_transformed_perrange,df_y_train],axis=1)
    test_name='./data_covid_smoted_DISCRETIZED_completedata'+'.txt'
    Xtotal.to_csv(test_name,sep=' ',index=False,header=False)
    #return X_transformed_perrange,df_y_train 



if __name__ == '__main__':
    
    main()



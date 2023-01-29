import scipy.stats as stats
import pandas as pd
import os







df = pd.read_csv("res.csv")
df.head()
print("Proposed model vs sigd2",stats.ttest_ind(df.dropna()['prop'], df.dropna()['Sigd2'])[1]) 
print("Proposed model vs random",stats.ttest_ind(df.dropna()['prop'], df.dropna()['rand'])[1] )
print("Proposed model vs RF",stats.ttest_ind(df.dropna()['prop'], df.dropna()['RF'])[1] )

print()
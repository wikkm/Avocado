
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import datetime
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("avocado.csv", index_col=[0])
X = data.iloc[:,:]  #independent columns
y = X['AveragePrice']    #target column i.e price range
X = X.drop(columns=['AveragePrice'])


############# converting all values to numbers #############
for index, row in X.iterrows():
    # converting dates to miliseconds
    date_time_obj = datetime.datetime.strptime(row['Date'], '%Y-%m-%d')
    X.at[index, 'Date'] = date_time_obj.timestamp()

        
# converting types to 0/1
labelsForType, uniqueForType = pd.factorize(X['type'])

X['type'] = labelsForType

# converting regions to numbers
labelsForRegion, uniqueForRegion = pd.factorize(X['region'])

X['region'] = labelsForRegion

####################################################

df = X.values
scaler = StandardScaler()
X = scaler.fit_transform(df)
y = y.values

names = ['Date',"Total Volume","PLU4046","PLU4225","PLU4770","Total Bags","Small Bags","Large Bags","XLarge Bags","type","year","region"]
lasso = Lasso(alpha=.0001)
lasso.fit(X, y)

def sortPrinted(coefs, names, sort):
    lst = zip(coefs, names)
    if sort:
        lst = sorted(lst,  key = lambda x:-np.abs(x[0]))
    return " + ".join("%s * %s" % (round(coef, 3), name)
                                   for coef, name in lst)

print ("Lasso model: ", sortPrinted(lasso.coef_, names, sort = True))


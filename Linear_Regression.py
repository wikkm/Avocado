import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from math import sqrt

df = pd.read_csv("Avocado.csv")
typeofAvocado = {'conventional': 1,'organic': 2} 
df.type = [typeofAvocado[item] for item in df.type]

# converting regions to numbers
labelsForRegion, uniqueForRegion = pd.factorize(df['region'])

df['region'] = labelsForRegion

df_arr = np.array(df)

x_4046 = df_arr[:, 4:5]
x_4225 = df_arr[:, 5:6]
x_4770 = df_arr[:, 6:7]

x_small = df_arr[:, 8:9]
x_large = df_arr[:, 9:10]
x_xlarge = df_arr[:, 10:11]

x_type = df_arr[:, 11:12]
x_year = df_arr[:, 12:13]
x_region = df_arr[:, 13:14]

y = df_arr[:, 2:3] # Prices
#Used to extend the linear regression line to fit the data set
x_line = [0,23000000]
y_line = []
#Number of splits done in k-fold
num_of_splits = 9
rmse_total = 0
#Setup the equations
kf = KFold(n_splits=num_of_splits)
reg = linear_model.LinearRegression()
fold = 0
#Loop through each fold
print('PLU: 4046')
for train_index, test_index in kf.split(x_4046):
    X_train, X_test = x_4046[train_index], x_4046[test_index]
    y_train, y_test = y[train_index], y[test_index]
    reg.fit(X_train, y_train)
    pred = reg.predict(X_test)
    y_line = [reg.intercept_, reg.coef_ * 23000000 + reg.intercept_]
    plt.scatter(X_train,y_train,color='red', label='Train', s=1)
    plt.scatter(X_test,y_test,color='blue', label='Test', s=1)
    plt.plot(x_line, y_line, color='black', linewidth=3) 
    plt.legend()
    plt.title('PLU: 4046')
    plt.xlabel('Volume')
    plt.ylabel('Price in $')
    plt.ylim(0, 3.5)
    plt.show()
    rmse = sqrt(mean_squared_error(y_test, pred))
    rmse_total = rmse_total + rmse
    print("Fold: {} | Coef: {} | Intercept: {} | RMSE: {}".format(fold, reg.coef_, reg.intercept_, rmse))
    fold = fold + 1
print("Avg RMSE: {}".format(rmse_total / num_of_splits))
fold = 0
print('PLU: 4225')
rmse_total = 0
for train_index, test_index in kf.split(x_4225):
    X_train, X_test = x_4225[train_index], x_4225[test_index]
    y_train, y_test = y[train_index], y[test_index]
    reg.fit(X_train, y_train)
    pred = reg.predict(X_test)
    y_line = [reg.intercept_, reg.coef_ * 5200000 + reg.intercept_]
    plt.scatter(X_train,y_train,color='red', label='Train', s=1)
    plt.scatter(X_test,y_test,color='blue', label='Test', s=1)
    plt.plot(x_line, y_line, color='black', linewidth=3) 
    plt.legend()
    plt.title('PLU: 4225')
    plt.xlabel('Volume')
    plt.ylabel('Price in $')
    plt.ylim(0, 3.5)
    plt.show()
    rmse = sqrt(mean_squared_error(y_test, pred))
    rmse_total = rmse_total + rmse
    print("Fold: {} | Coef: {} | Intercept: {} | RMSE: {}".format(fold, reg.coef_, reg.intercept_, rmse))
    fold = fold + 1
print("Avg RMSE: {}".format(rmse_total / num_of_splits))

x_line = [0,2600000]
fold = 0
print('PLU: 4770')
rmse_total = 0
for train_index, test_index in kf.split(x_4770):
    X_train, X_test = x_4770[train_index], x_4770[test_index]
    y_train, y_test = y[train_index], y[test_index]
    reg.fit(X_train, y_train)
    pred = reg.predict(X_test)
    y_line = [reg.intercept_, reg.coef_ * 2600000 + reg.intercept_]
    plt.scatter(X_train,y_train,color='red', label='Train', s=1)
    plt.scatter(X_test,y_test,color='blue', label='Test', s=1)
    plt.plot(x_line, y_line, color='black', linewidth=3) 
    plt.legend()
    plt.title('PLU: 4770')
    plt.xlabel('Volume')
    plt.ylabel('Price in $')
    plt.ylim(0, 3.5)
    plt.show()
    rmse = sqrt(mean_squared_error(y_test, pred))
    rmse_total = rmse_total + rmse
    print("Fold: {} | Coef: {} | Intercept: {} | RMSE: {}".format(fold, reg.coef_, reg.intercept_, rmse))
    fold = fold + 1
print("Avg RMSE: {}".format(rmse_total / num_of_splits))

fold = 0
print('Small bags')
x_line = [0,14000000]
rmse_total = 0
for train_index, test_index in kf.split(x_small):
    X_train, X_test = x_small[train_index], x_small[test_index]
    y_train, y_test = y[train_index], y[test_index]
    reg.fit(X_train, y_train)
    pred = reg.predict(X_test)
    y_line = [reg.intercept_, reg.coef_ * 14000000 + reg.intercept_]
    plt.scatter(X_train,y_train,color='red', label='Train', s=1)
    plt.scatter(X_test,y_test,color='blue', label='Test', s=1)
    plt.plot(x_line, y_line, color='black', linewidth=3) 
    plt.legend()
    plt.title('Small bags')
    plt.xlabel('Volume')
    plt.ylabel('Price in $')
    plt.ylim(0, 3.5)
    plt.show()
    rmse = sqrt(mean_squared_error(y_test, pred))
    rmse_total = rmse_total + rmse
    print("Fold: {} | Coef: {} | Intercept: {} | RMSE: {}".format(fold, reg.coef_, reg.intercept_, rmse))
    fold = fold + 1
print("Avg RMSE: {}".format(rmse_total / num_of_splits))

fold = 0
print('Large bags')
x_line = [0,5800000]
rmse_total = 0
for train_index, test_index in kf.split(x_large):
    X_train, X_test = x_large[train_index], x_large[test_index]
    y_train, y_test = y[train_index], y[test_index]
    reg.fit(X_train, y_train)
    pred = reg.predict(X_test)
    y_line = [reg.intercept_, reg.coef_ * 5800000 + reg.intercept_]
    plt.scatter(X_train,y_train,color='red', label='Train', s=1)
    plt.scatter(X_test,y_test,color='blue', label='Test', s=1)
    plt.plot(x_line, y_line, color='black', linewidth=3) 
    plt.legend()
    plt.title('Large bags')
    plt.xlabel('Volume')
    plt.ylabel('Price in $')
    plt.ylim(0, 3.5)
    plt.show()
    rmse = sqrt(mean_squared_error(y_test, pred))
    rmse_total = rmse_total + rmse
    print("Fold: {} | Coef: {} | Intercept: {} | RMSE: {}".format(fold, reg.coef_, reg.intercept_, rmse))
    fold = fold + 1
print("Avg RMSE: {}".format(rmse_total / num_of_splits))

fold = 0
print('X-Large bags')
x_line = [0,560000]
rmse_total = 0
for train_index, test_index in kf.split(x_xlarge):
    X_train, X_test = x_xlarge[train_index], x_xlarge[test_index]
    y_train, y_test = y[train_index], y[test_index]
    reg.fit(X_train, y_train)
    pred = reg.predict(X_test)
    y_line = [reg.intercept_, reg.coef_ * 560000 + reg.intercept_]
    plt.scatter(X_train,y_train,color='red', label='Train', s=1)
    plt.scatter(X_test,y_test,color='blue', label='Test', s=1)
    plt.plot(x_line, y_line, color='black', linewidth=3) 
    plt.legend()
    plt.title('X-Large bags')
    plt.xlabel('Volume')
    plt.ylabel('Price in $')
    plt.ylim(0, 3.5)
    plt.show()
    rmse = sqrt(mean_squared_error(y_test, pred))
    rmse_total = rmse_total + rmse
    print("Fold: {} | Coef: {} | Intercept: {} | RMSE: {}".format(fold, reg.coef_, reg.intercept_, rmse))
    fold = fold + 1
print("Avg RMSE: {}".format(rmse_total / num_of_splits))

fold = 0
x_line = [0,3]
print('Type')
rmse_total = 0
for train_index, test_index in kf.split(x_type):
    X_train, X_test = x_type[train_index], x_type[test_index]
    y_train, y_test = y[train_index], y[test_index]
    reg.fit(X_train, y_train)
    pred = reg.predict(X_test)
    y_line = [reg.intercept_, reg.coef_ * 3 + reg.intercept_]
    plt.scatter(X_train,y_train,color='red', label='Train', s=1)
    plt.scatter(X_test,y_test,color='blue', label='Test', s=1)
    plt.plot(x_line, y_line, color='black', linewidth=3) 
    plt.legend()
    plt.title('Type')
    plt.xlabel('Type')
    plt.ylabel('Price in $')
    plt.ylim(0, 3.5)
    plt.show()
    rmse = sqrt(mean_squared_error(y_test, pred))
    rmse_total = rmse_total + rmse
    print("Fold: {} | Coef: {} | Intercept: {} | RMSE: {}".format(fold, reg.coef_, reg.intercept_, rmse))
    fold = fold + 1
print("Avg RMSE: {}".format(rmse_total / num_of_splits))

fold = 0
x_line = [0,2019]
print('Year')
rmse_total = 0
for train_index, test_index in kf.split(x_year):
    X_train, X_test = x_year[train_index], x_year[test_index]
    y_train, y_test = y[train_index], y[test_index]
    reg.fit(X_train, y_train)
    pred = reg.predict(X_test)
    y_line = [reg.intercept_, reg.coef_ * 2019 + reg.intercept_]
    plt.scatter(X_train,y_train,color='red', label='Train', s=1)
    plt.scatter(X_test,y_test,color='blue', label='Test', s=1)
    plt.plot(x_line, y_line, color='black', linewidth=3) 
    plt.legend()
    plt.title('Year')
    plt.xlabel('Year')
    plt.ylabel('Price in $')
    plt.xlim(2014,2019)
    plt.ylim(0, 3.5)
    plt.show()
    rmse = sqrt(mean_squared_error(y_test, pred))
    rmse_total = rmse_total + rmse
    print("Fold: {} | Coef: {} | Intercept: {} | RMSE: {}".format(fold, reg.coef_, reg.intercept_, rmse))
    fold = fold + 1
print("Avg RMSE: {}".format(rmse_total / num_of_splits))

fold = 0
print('Region')
x_line = [0,54]
rmse_total = 0
for train_index, test_index in kf.split(x_region):
    X_train, X_test = x_region[train_index], x_region[test_index]
    y_train, y_test = y[train_index], y[test_index]
    reg.fit(X_train, y_train)
    pred = reg.predict(X_test)
    y_line = [reg.intercept_, reg.coef_ * 51 + reg.intercept_]
    plt.scatter(X_train,y_train,color='red', label='Train', s=1)
    plt.scatter(X_test,y_test,color='blue', label='Test', s=1)
    plt.plot(x_line, y_line, color='black', linewidth=3) 
    plt.legend()
    plt.title('Region')
    plt.xlabel('Region')
    plt.ylabel('Price in $')
    plt.ylim(0, 3.5)
    plt.show()
    rmse = sqrt(mean_squared_error(y_test, pred))
    rmse_total = rmse_total + rmse
    print("Fold: {} | Coef: {} | Intercept: {} | RMSE: {}".format(fold, reg.coef_, reg.intercept_, rmse))
    fold = fold + 1
print("Avg RMSE: {}".format(rmse_total / num_of_splits))
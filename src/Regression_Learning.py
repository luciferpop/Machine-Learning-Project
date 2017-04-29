'''
Created on Nov 8, 2015

@author: Lihua
'''
from scipy.constants.constants import alpha
from sklearn.linear_model.coordinate_descent import ElasticNet, ElasticNetCV
print(__doc__)

import re
import copy as cp
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

SFPD_TEST       = 'SFPD_Incidents_Test.csv'
SFPD_TRAIN      = 'SFPD_Incidents_-_Current_year__2015_.csv'
SFPD_TRAIN_10K   = 'SFPD_Incidents_10k.csv'
SFPD_TRAIN_5K   = 'SFPD_Incidents_5k.csv'
SFPD_TRAIN_PLT  = 'SFPD_Incidents_2015_Train_Small.csv'

# data from file named 'filename'
def get_data(filename):
    category, dow, time, resolution = np.loadtxt(filename, delimiter=',', 
                                usecols=(1, 3, 5, 7), dtype='str', 
                                skiprows = 1, unpack=True)
    x, y           = np.loadtxt(filename, delimiter=',',
                                usecols=(9, 10), dtype='float',
                                skiprows = 1, unpack=True)
    return category, dow, time, resolution, x, y

# retrieve numbers from time and save it into array at
def convert_time(time):
    t, at = [], []
    for item in time:
        at = np.append(at, re.findall(r'(\w*[0-9]+)\w*', item))
            
    # convert string to float and save in into res
    at = [float(item) for item in at]
    for i in range(len(at)/2):
        t.append(at[2*i] + at[2*i+1]/60.)
    return t

# normalize the original data
def scale(X):
    x = cp.copy(X)
    X_mean = np.mean(x)
    X_Std = np.std(x)
    for i in range(len(x)):
        x[i] = (x[i]-X_mean) / X_Std
    return x

# merge two/three feature arrays into one array
def merge(x1, x2, x3, x4, x5):
    merged = []
    for i in range(len(x1)):
        temp = []
        temp.append(x1[i])
        temp.append(x2[i])
        temp.append(x3[i])
        temp.append(x4[i])
        temp.append(x5[i])
        merged.append(temp)
    return merged

# assign an unique ID to each training and tesing data 
def assign(data, data_):
    id = []
    id_ = []
    table = list(set(data))
    for item in data:
        id.append(table.index(item))
    for item in data_:
        id_.append(table.index(item))
    return table, id, id_
    
# plot ridge trace from 10^-10 to 10^3
def plt_ridge_trace(X, y):
    n_alpha = 200
    alphas = np.logspace(-10, 6, n_alpha)
    clf = linear_model.Ridge(fit_intercept=False)

    coefs = []
    for a in alphas:
        clf.set_params(alpha=a)
        clf.fit(X, y)
        coefs.append(clf.coef_)
    ax = plt.gca()
    ax.set_color_cycle(['r', 'y', 'g', 'b', 'm'])
    ax.plot(alphas, coefs)
    ax.set_xscale('log')    
    ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
    plt.xlabel('lambda')
    plt.ylabel('weights')
    plt.title('Ridge trace of differen lambdas')
    plt.show()
    
# print error metrics 
def print_err(clf, X_, y_):
    # (1 - u/v), where u is the regression sum of squares ((y_true - y_pred) ** 2).sum() 
    # and v is the residual sum of squares ((y_true - y_true.mean()) ** 2).sum()
    print('Score: %.4f' 
          % clf.score(X_, y_))
    # the mean of squared error
    print("Mean of squared residual: %.4f"
      % np.mean((clf.predict(X_) - y_)**2))

# Ordinary Least Squares Regression
def OLS_regre(X, y, X_, y_):
    print 'Ordinary Least Squares'
    clf = linear_model.LinearRegression()
    clf.fit(X, y)
    print_err(clf, X_, y_)
    return clf

# Ridge Regression
def ridge_regre(X, y, X_, y_):
    print 'Ridge Regression'
    clf = linear_model.Ridge()
    clf.set_params(alpha=2)
    clf.fit(X, y)
    print_err(clf, X_, y_)
    return clf

# Elastic net
def enet(X, y, X_, y_):
    print 'Elastic Net'
    clf = ElasticNet(alpha=0.0001, l1_ratio=1)
    clf.fit(X, y)
    print_err(clf, X_, y_)
    return clf

# Prediction 
def predict(clf, X_test):
    y_predict = clf.predict(X_test)
    for id in y_predict:
        print round(id)

# plot an exmple of OLS regression
def plt_ex(X_, y, X_test, y_test, clf):
    plt.scatter(X_test, y_test, color='blue')
    y = clf.intercept_ + clf.coef_*X_
    plt.plot(X_, y, color='r')
    plt.xticks(())
    plt.yticks(())
    plt.show()
    
def main():
    # retrieve data from training and testing examples
    train_cate, train_dow, train_time, train_res, train_x, train_y = get_data(SFPD_TRAIN_10K)
    test_cate, test_dow, test_time, test_res, test_x, test_y = get_data(SFPD_TEST)
     
    # covert time to the numeric formation
    # get unique ID for different category 
    # save code array 'cate'
    train_time   = convert_time(train_time)
    test_time    = convert_time(test_time)

    # covert features to numberical parameters
    cate_table, train_cate_id, test_cate_id = assign(train_cate, test_cate)
    dow_table, train_dow_id, test_dow_id = assign(train_dow, test_dow)
    res_table, train_res_id, test_res_id = assign(train_res, test_res)
    
    # prepare training and testing data
    X = np.array(merge(train_dow_id, train_time, train_res_id, train_x, train_y))
    y = np.array(train_cate_id)
    X_ = np.array(merge(test_dow_id, test_time, test_res_id, test_x, test_y))
    y_ = np.array(test_cate_id)

    # apply different regression algorithms
    Xt = train_time
    X_t = test_time
    clf = OLS_regre(Xt, y, X_t, y_)

if __name__ == '__main__':
    main()
'''
    X_x = test_x
    X_ = [[0 for j in range(1)] for i in range(len(train_x))]
    for i in range(len(train_x)):
        for j in range(1):
            X_[i][j] = train_x[i]
    
    clf = linear_model.LinearRegression()
    clf.fit(X_, y)   
    plt_ex(X_, y, X_x, y_test, clf)
'''


    

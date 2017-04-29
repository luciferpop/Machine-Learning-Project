import re
import time
import numpy as np
import copy as cp
from random import uniform
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

FILE_LENGTH     = 13

LOG             = 'log.log'
ACCURACY        = 'accuracy.log'
SFPD_TEST       = 'SFPD_Incidents_Test.csv'
SFPD_TRAIN      = 'SFPD_Incidents_-_2015.csv'
SFPD_TRAIN_TEST = 'SFPD_Incidents_10k.csv'

# retrieve data from file named 'filename'
def get_data(filename):
    category, time = np.loadtxt(filename, delimiter=',', 
                                usecols=(1, 5), dtype='str', 
                                skiprows = 1, unpack=True)
    x, y           = np.loadtxt(filename, delimiter=',',
                                usecols=(9, 10), dtype='float',
                                skiprows = 1, unpack=True)
    return category, time, x, y

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

def CombineColumns(time, x, y, len):
    trainData = []
    for i in range(len):
        featureVector = []
        featureVector.append(time[i])
        featureVector.append(x[i])
        featureVector.append(y[i])
        trainData.append(featureVector)
    return trainData

# Gaussian Bayes Prediction
def gau_nb(X, y):
    classifier=GaussianNB()
    classifier.fit(X, y)
    return classifier


# Multinomial Bayes Prediction
def mul_nb(X, y):
    classifier= MultinomialNB()
    classifier.fit(X, y)
    MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
    return classifier

# print predictions 
def get_pre(cate_table, X_, y_, classifier):
    predicted = []
    len_cate, len_test = len(cate_table), len(y_)
    
    proba =  classifier.predict_proba(X_).tolist()
    proba_ = cp.deepcopy(proba)
    
    for i in range(len_test):
        temp = []
        proba_[i].sort()
        for j in range(len_cate-1, len_cate-5, -1):
            k = proba[i].index(proba_[i][j])
            temp.append(cate_table[k])
            proba[i][k] = proba[i][k] - uniform(0, 1)
        predicted.append(temp)
        
    return predicted

# print accuracy
def print_acc(test_cate, predicted):
    count = .0
    length = len(test_cate)
    for i in range(length):
        if test_cate[i] in predicted[i]:
            #print 'Incident No.: %d, Target: %s' %(i, test_cate[i])
            count = count + 1
            #for j in range(4):
                #print '[%d] %s' %(j, predicted[i][j]) 
            #print '-------------------'
    #print 'Total Accuracy: %f' %(count / length)
    return count / length

# main
def main():
    flog      = open(LOG, 'a')
    faccuracy = open(ACCURACY, 'a')
    test_cate, test_time, test_x, test_y = get_data(SFPD_TEST)
    start = time.time()
    SFPD_TRAIN = 'SFPD_Incidents_-_%s.csv' %(2004)
    train_cate, train_time, train_x, train_y = get_data(SFPD_TRAIN)
        
    # covert time to the numeric formation
    # get unique ID for different category 
    # save code array 'cate'
    train_time   = convert_time(train_time)
    test_time = convert_time(test_time)
       
    # assign id and get table from data
    cate_table, train_cate_id, test_cate_id = assign(train_cate, test_cate)
       
    # prepare training data and testing data
    X = CombineColumns(train_time, train_x, train_y, len(train_time))
    y = train_cate_id
        
    X_ = CombineColumns(test_time, test_x, test_y, len(test_time))
    y_ = test_cate_id
       
    # using Gaussian Naive Bayes Prediction
    clf = gau_nb(X, y)
    predicted = get_pre(cate_table, X_, y_, clf)
    flog.write('%s IS FINISHED.\tTIME CONSUMED: %.2fmin\t ACCURACY: %f\n' 
            %(SFPD_TRAIN, (time.time() - start)/60, print_acc(test_cate, predicted)))
    faccuracy.write('%f\n' %(print_acc(test_cate, predicted)))
    print '%s IS FINISHED.' %SFPD_TRAIN
    flog.close()
    faccuracy.close()
if __name__ == '__main__':
    main()
'''
#     for i in range(FILE_LENGTH):
#         start = time.time()
#         SFPD_TRAIN = 'SFPD_Incidents_-_%s.csv' %(i+2003)
#         train_cate, train_time, train_x, train_y = get_data(SFPD_TRAIN)
#             
#         # covert time to the numeric formation
#         # get unique ID for different category 
#         # save code array 'cate'
#         train_time   = convert_time(train_time)
#         test_time = convert_time(test_time)
#            
#         # assign id and get table from data
#         cate_table, train_cate_id, test_cate_id = assign(train_cate, test_cate)
#            
#         # prepare training data and testing data
#         X = CombineColumns(train_time, train_x, train_y, len(train_time))
#         y = train_cate_id
#             
#         X_ = CombineColumns(test_time, test_x, test_y, len(test_time))
#         y_ = test_cate_id
#            
#         # using Gaussian Naive Bayes Prediction
#         clf = gau_nb(X, y)
#         predicted = get_pre(cate_table, X_, y_, clf)
#         flog.write('%s IS FINISHED.\tTIME CONSUMED: %.2fmin\t ACCURACY: %f\n' 
#                 %(SFPD_TRAIN, (time.time() - start)/60, print_acc(test_cate, predicted)))
#         faccuracy.write('%f\n' %(print_acc(test_cate, predicted)))
#         print '%s IS FINISHED.' %SFPD_TRAIN
'''
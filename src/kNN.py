'''
Created on Oct 6, 2015

@author: Lihua
'''
print(__doc__)

import re
import copy as cp
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from random import uniform

SFPD_TEST       = 'SFPD_Incidents_Test.csv'
SFPD_TRAIN = 'SFPD_Incidents_-_2015.csv'

# 1st, 2nd, 3rd, 4th, and 5th row represents 
# 'SEXUAL CRIME', 'FINANCIAL CRIME', 'THEFT/BRIBERY', 'ALCOHO/DRUG', and 'VIOLENCE' respectively 
CATE_TABLE = ['PORNOGRAPHY/OBSCENE MAT', 'PROSTITUTION', 'SEX OFFENSES/ FORCIBLE', 'SEX OFFENSES/ NON FORCIBLE',
              'BAD CHECKS', 'FORGERY/COUNTERFEITING', 'EXTORTION', 'FRAUD', 'GAMBLING', 'EMBEZZLEMENT',
              'BRIBERY', 'BURGLARY', 'LARCENY/THEFT', 'ROBBERY', 'STOLEN PROPERTY', 'VEHICLE THEFT',
              'DRIVING UNDER THE INFLUENCE', 'DRUG/NARCOTIC', 'DRUNKENNESS', 'LIQUOR LAWS',
              'ARSON', 'ASSAULT', 'FAMILY OFFENSES', 'KIDNAPPING', 'SUICIDE', 'TRESPASS',
              'DISORDERLY CONDUCT', 'LOITERING', 'MISSING PERSON', 'NON-CRIMINAL', 'OTHER OFFENSES', 'RUNAWAY', 'SECONDARY CODES', 'SUSPICIOUS OCC', 'TREA', 'VANDALISM', 'WARRANTS', 'WEAPON LAWS']

TYPE = ['SEXUAL CRIME', 'FINANCIAL CRIME', 'THEFT/BRIBERY', 'ALCOHO/DRUG', 'VIOLENCE']

# retrieve data from file named 'filename'
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

# normalize the original data
def scale(X):
    x = cp.copy(X)
    X_mean = np.mean(x)
    X_Std = np.std(x)
    for i in range(len(x)):
        x[i] = (x[i]-X_mean) / X_Std
    return x

# calculating Euclidean distances between training and testing examples
# x1, x2, x3, x4, and x5 represents day of week, time, resolution, x coordinate, and y coordinate, respectively 
# x and x_ represents training and testing examples, respectively 
def cal_dist(x1, x2, x3, x4, x5, x1_, x2_, x3_, x4_, x5_):
    dist = [[0 for j in range(len(x1))] for i in range(len(x1_))]
    for i in range(len(x1_)):
        for j in range(len(x1)):
            dist[i][j] = np.sqrt((x1[j] - x1_[i])**2 + (x2[j] - x2_[i])**2 
                                 + (x3[j] - x3_[i])**2 + (x4[j] - x4_[i])**2 + (x5[j] - x5_[i])**2) 
    return dist
        
#kNN algorithm
def kNN(dist_, k, len_test):
    temp = 0
    index = [[0 for j in range(k)] for i in range(len_test)]
    for i in range(len_test):
        dist_copy = cp.copy(dist_[i])
        dist_copy.sort()
        for j in range(k):
            temp = dist_[i].index(dist_copy[j])
            index[i][j] = temp
            dist_[i][temp] = j + uniform(1, 10)
    return index

# make a prediction
def predict(k, len_test, index, train_cate, test_cate):
    # cat and cnt store category and count respectively
    cat = [[0 for j in range(k)] for i in range(len_test)]
    cnt = [[0 for j in range(k)] for i in range(len_test)]
    cate = [[0 for j in range(k)] for i in range(len_test)]
    
    for i in range(len_test):
        for j in range(k):
            cate[i][j] = train_cate[index[i][j]]
    
    for i in range(len_test):
        j = 0
        for l, m in Counter(cate[i]).iteritems():
            #print l, m
            cat[i][j] = l
            cnt[i][j] = m
            j = j + 1
        #print '------------'
    
    return cat, cnt

# get prediction accuracy 
def get_accracy(tcate, cat, cnt):
    # store those indices which match with targets
    id = []
    count = .0
    for i in range(len(tcate)):
        if tcate[i] in cat[i]:
            j = 0
            id.append(i)
            #print 'Incident No.: %d, Target: %s' %(i, tcate[i])
            for item in cat[i]:
                if item != 0:
                    print '[%d] %s' %(j, item)
                    j = j + 1
            count = count + 1
            #print '--------------------'
    #print 'Total Accuracy: %s' %(count / len(tcate))
    return count / len(tcate)

 # rearrange category and count 
def rearrange(k, cat, cnt, len_test):
    cat_ = [[0 for j in range(k)] for i in range(len_test)]
    cnt_ = [[0 for j in range(k)] for i in range(len_test)]
    for i in range(len_test):
        for j in range(len(cat)):
            # belongs to 'SEXUAL CRIME'
            if (CATE_TABLE.index(cat[i][j]) <= 3):
                cat_[i][j] = 'SEXUAL CRIME'
        
# draw an example
def draw(cat, cnt):
    cate, count = [], []
    for i in range(len(cat)):
        if cat[i] != 0:
            cate.append(cat[i])
    for j in range(len(cnt)):
        if cnt[j] != 0:
            count.append(cnt[j])
    print cate, count        
    colors = ['r', 'w', 'b', 'cyan', 'gold']    
    explode = (0, 0, 0, 0.1, 0)    
    plt.pie(count, labels=cate, autopct='%1.1f%%', explode=explode, colors=colors, shadow=True, startangle=90)
    plt.axis('equal')
    plt.show()

# calculate weight according to weighted-knn algorithm
def get_weight(k, len_test, cnt, dist, index, train_cate):
    weight = [0 for j in range(k) for i in range(len_test)]
    for i in range(len_test):
        for l, m in Counter(index[i]).iteritems():
            print dist[i][l], train_cate[l]
        print '-----------------'
            
#     for i in range(len_test):
#         for j in range(k):
#             weight[i][j] = dist[i][index[i][j]]

# main
def main():
    # retrieve data from training and testing examples
    train_cate, train_dow, train_time, train_res, train_x, train_y = get_data(SFPD_TRAIN)
    test_cate, test_dow, test_time, test_res, test_x, test_y = get_data(SFPD_TEST)
    tcate = test_cate
     
    # covert time to the numeric formation
    train_time   = convert_time(train_time)
    test_time = convert_time(test_time)
    
    # assign id and get table from data
    cate_table, train_cate_id, test_cate_id = assign(train_cate, test_cate)
    dow_table, train_dow_id, test_dow_id = assign(train_dow, test_dow)
    res_table, train_res_id, test_res_id = assign(train_res, test_res)
    
    x1 = train_dow_id
    x2 = scale(train_time)
    x3 = train_res_id
    x4 = scale(train_x)
    x5 = scale(train_y)
    
    x1_ = test_dow_id
    x2_ = scale(test_time)
    x3_ = test_res_id
    x4_ = scale(test_x)
    x5_ = scale(test_y)
       
    len_test = len(test_time)
    dist = cal_dist(x1, x2, x3, x4, x5, x1_, x2_, x3_, x4_, x5_)
    dist_ = cp.deepcopy(dist)
    index = kNN(dist_, 9, len_test)
    cat, cnt = predict(9, len_test, index, train_cate, test_cate)
    get_accracy(tcate, cat, cnt)
    
if __name__ == '__main__':
    main()


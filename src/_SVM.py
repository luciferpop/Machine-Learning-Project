'''
Created on Nov 5, 2015

@author: Lihua
'''
print(__doc__)

import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

SFPD_TEST       = 'SFPD_Incidents_Test.csv'
SFPD_TRAIN      = 'SFPD_Incidents_-_Current_Year__2015_.csv'
SFPD_TRAIN_5K   = "SFPD_Incidents_5k.csv"
SFPD_TRAIN_PLT  = 'SFPD_Incidents_2015_Train_Small.csv'

# retrieve data from filename
def get_data(filename):
    category, time = np.loadtxt(filename, delimiter=',', 
                                usecols=(1, 5,), dtype='str', 
                                skiprows = 1, unpack=True)
    x, y           = np.loadtxt(filename, delimiter=',',
                                usecols=(9, 10), dtype='float',
                                skiprows = 1, unpack=True)
    return category, time, x, y

#retrieve numbers from time and save it into array at
def convert_time(time):
    t, at = [], []
    for item in time:
        at = np.append(at, re.findall(r'(\w*[0-9]+)\w*', item))
            
    # convert string to float and save in into res
    at = [float(item) for item in at]
    for i in range(len(at)/2):
        t.append(at[2*i] + at[2*i+1]/60.)
    return t

# merge two feature arrays into one array
def merge(x1, x2, k):
    crd = [[0 for j in range(k)] for i in range(len(x1))]
    for i in range(len(x1)):
        for j in range(k):
            if (j==0):
                crd[i][j] = x1[i]
            else:
                crd[i][j] = x2[i]
    return crd

# assign an unique ID to each category 
def assign(train_cate):
    cate_ID = []
    cate = train_cate
    cate = list(set(train_cate))
    cate.sort()
    for item in train_cate:
        cate_ID.append(cate.index(item))
    return cate, cate_ID

# plot different SVM classifiers 
def plot(crd, cate_ID, clf):
    h = 0.02
    # create a mesh to plot in
    x_min, x_max = crd[:, 0].min() - 1, crd[:, 0].max() + 1
    y_min, y_max = crd[:, 1].min() - 1, crd[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # plot the reslut
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    # plot also the training points
    plt.scatter(crd[:, 0], crd[:, 1], c=cate_ID, cmap=plt.cm.Paired)
    
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.show()
    
# main
def main():
    # retrieve data from training and testing examples
    train_cate, train_time, train_x, train_y = get_data(SFPD_TRAIN_5K)
    test_cate, test_time, test_x, test_y = get_data(SFPD_TEST)
     
    # covert time to the numeric formation
    train_time   = convert_time(train_time)
    test_time = convert_time(test_time)
    
    # get category table and ID, 
    cate, cate_ID = assign(train_cate)
    
    # merge x and y coordinates
    crd = merge(train_x, train_y, 2)
    np_crd = np.array(crd)
    np_cate = np.array(cate_ID)
    
    svc = svm.SVC(kernel='linear', C=1.).fit(np_crd, np_cate)
    print svc.support_vectors_.shape
    
       
   
   
   
   
 
 
    
if __name__ == '__main__':
    main()
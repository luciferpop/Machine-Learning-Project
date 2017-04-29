import numpy as np
import copy as cp
from random import uniform

from sklearn.naive_bayes import GaussianNB
def main():
    
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2], [4, 7], [7, 9]])
    Y = np.array([1, 1, 1, 2, 3, 4, 4, 4])
    
    clf = GaussianNB()
    clf.fit(X, Y)
    proba = clf.predict_proba([[-0.8, -1]]).tolist()
    proba_ = cp.deepcopy(proba) 
    
    print proba[0]
    
    for i in range(1):
        proba_[i].sort()
        for j in range(3, 1, -1):
            index = proba[i].index(proba_[i][j])
            proba[i][index] = proba[i][index] - 1
            index = index + 1
            print index
    
    print clf.predict([[-0.8, -1]])  
    print uniform(0 ,1)  
main()

import numpy as np
from alogrithm import *


class ID3(Alogrothm):

    def __init__(self,depth =None,min_split =None):
        self._X = {}
        self._Y = {}
        self._tree ={}
        self._max_depth = depth
        self._min_split = min_split
        pass
    def entropy(self,y):
        #calculate entropy
        if len(y) <=1:
            return 0
        totalcount = np.bincount(y)
        prob = totalcount[np.nonzero(totalcount)]/len(y)
        if len(prob)<=1:
            return 0
        return -np.sum(prob*np.log2(prob))
    def info_gain(self,prev_y , current_y):
        #caluculate the info gain
        cond_entopy =0
        for y in current_y:
            cond_entopy += self.entropy(y)*len(y)/len(prev_y)
        info_gain = self.entropy(prev_y)-cond_entopy
        return info_gain
    def divide(self,X,Y,split_value,split_atr):
        #divide the data into branches
        cloumn_split = X[:,split_atr]
        if split_value is None or (isinstance(split_value,float) and np.isnan(split_value)):
            return X[:0], Y[:0], X, Y
        if isinstance(split_value , (int, float,np.integer,np.floating)):
            mask = cloumn_split > split_value
        else:
            mask = cloumn_split == split_value
        X_right,Y_right = X[mask],Y[mask]
        X_left,Y_left = X[~mask],Y[~mask]
        return X_right ,Y_right ,X_left ,Y_left
    def find_split_val(self,X,Y,split_atr):
        #divide based on the best value where you gain the most info
        best_info_gain = 0
        col_split = self._unique_vals[split_atr]
        best_split_value = col_split[0]
        for i in col_split:
            if i is None or (isinstance(i, float) and np.isnan(i)):
                continue
            X_right ,Y_right,X_left ,Y_left  = self.divide(X,Y,i,split_atr)
            current_y = []
            current_y.append(Y_left)
            current_y.append(Y_right)
            info_gain = self.info_gain(Y, current_y)
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_split_value = i
        return best_split_value,best_info_gain
    def find_split_atr(self,X,Y):
        #divide based on the best atr where you gain tthe most info
        best_info_gain = 0
        best_split_atr = 0
        for i in range(X.shape[1]):
            split_val,info_gain = self.find_split_val(X,Y,i)
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_split_atr = i
        return best_split_atr,best_info_gain
    def majority_class(self,y):
        values, counts = np.unique(y, return_counts=True)
        return int(values[np.argmax(counts)])
    def recursive(self,X,Y,depth =0):
        #puting eveerthing together and creating the tree
        if len(Y) == 0:
            return {"Leaf": True, "Class": 0}
        if self.entropy(Y) ==0:
            return{"Leaf":True,"Class":int(Y[0])}
        if self._max_depth != None:
            if depth >= self._max_depth:
                return{"Leaf":True ,"Class":self.majority_class(Y)}
        if self._min_split != None:
            if len(Y)< self._min_split:
                return {"Leaf": True, "Class": self.majority_class(Y)}
        best_atr, best_info = self.find_split_atr(X, Y)
        best_val, best_info = self.find_split_val(X,Y,best_atr)
        if best_info <= 0:
            return {"Leaf": True, "Class": self.majority_class(Y)}
        X_right, Y_right, X_left, Y_left = self.divide(X,Y,best_val, best_atr)
        return{
                "Leaf":False,
                "Atr":best_atr,
                "Val":best_val,
                "Left_size":len(X_left),
                "Right_size":len(X_right),
                "Left":self.recursive(X_left,Y_left,depth+1),
                "Right":self.recursive(X_right,Y_right,depth+1),
            }

    def fit(self,X,z):
        self._X=np.array(X)
        self._Y=np.array([int(y)for y in z])
        self._unique_vals = {
            i: np.unique(self._X[:, i])
            for i in range(self._X.shape[1])
        }
        self._tree = self.recursive(self._X, self._Y)
    def predictone(self,x,tree):
        #go down the tree to predict the value
        if tree["Leaf"]:
            return tree["Class"]
        if x[tree["Atr"]] is None or isinstance(x[tree["Atr"]],float) and np.isnan(x[tree["Atr"]]):
            left_pred = self.predictone(x , tree["Left"])
            right_pred = self.predictone(x , tree["Right"])
            left_size = tree["Left_size"]
            right_size = tree["Right_size"]
            total_size = left_size + right_size
            left_weight = left_size / total_size
            right_weight = right_size / total_size
            if left_weight >= right_weight:
                return left_pred
            else:
                return right_pred
        if x[tree["Atr"]]>tree["Val"]:
            return self.predictone(x,tree["Right"])
        else:
            return self.predictone(x,tree["Left"])
    def predict(self,X):
        return [self.predictone(x, self._tree) for x in X]

    def check_accuracy(self, X, z):
        pred = self.predict(X)
        correct = sum(int(p) == int(actual) for p, actual in zip(pred, z))
        return (correct / len(X)) * 100


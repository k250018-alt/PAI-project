from IB3 import *
import numpy as np
class Randomforest(ID3):
    def __init__(self,min_split,depth,N_trees=50,max_features=.7):
        super().__init__(depth,min_split)
        self.__trees =[]
        self.__N_trees = N_trees
        self.__oob = []
        self.__max_features = max_features
        self.__bootstraps_row = []
        self.__bootstraps_col = []
    def _Bagging(self,X,num_features):
        #randomly pick data and the features
        x = list(range(X))
        row = np.random.choice(x,X)
        feat_count = max(1, int(num_features * self.__max_features))
        col = np.random.permutation(num_features)[:feat_count]
        return row,col
    def Bagging(self,X,num_features):
        #store the oob and the row and col indices
        for i in range(self.__N_trees):
            total = set(list(range(X)))
            row,col = self._Bagging(X,num_features)
            total = total -set(row)
            self.__bootstraps_row.append(row)
            self.__bootstraps_col.append(col)
            self.__oob.append(total)
    def oob(self,X,Y):
        #calcculate the oob score
        acuracy = []
        for i in range(len(X)):
            predictions =[]
            for j in range(self.__N_trees):
                if i in self.__oob[j]:
                    predictions.append(self.__trees[j].predict([X[i][self.__bootstraps_col[j]]])[0])
            if len(predictions)>0:
                acuracy.append(np.sum(np.array(predictions) == Y[i])/float(len(predictions)))
        return np.mean(acuracy)
    def predict(self,X):
        temp = []
        X =np.array(X)
        n= len(X)
        for i in range(self.__N_trees):
            tree = self.__trees[i]
            cols =self.__bootstraps_col[i]
            temp.append(tree.predict(X[:,cols]))
        temp =np.array(temp)
        pred =[]
        for i in range(n):
            values ,counts = np.unique(temp[:,i],return_counts=True)
            pred.append(int(values[np.argmax(counts)]))
        return pred
    def fit(self,X,Y):
        #craete n trees
        num_training,num_features = X.shape
        self.Bagging(num_training,num_features)
        for i in range(self.__N_trees):
            current_row,current_col = self.__bootstraps_row[i],self.__bootstraps_col[i]
            current_X = X[current_row[:,np.newaxis],current_col]
            current_Y = Y[current_row]
            model = ID3(self._max_depth,self._min_split)
            model.fit(current_X,current_Y)
            self.__trees.append(model)





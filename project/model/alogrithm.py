from abc import ABC, abstractmethod
class Alogrothm:
    @abstractmethod
    def fit(self,X,z):
        pass
    @abstractmethod
    def predict(self,X):
        pass
    @abstractmethod
    def check_accuracy(self,X,z):
        pass



from abc import ABCMeta, abstractmethod
import numpy as np
from sklearn.metrics import mean_absolute_error

class Model(object, metaclass=ABCMeta):

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def score(self, *args):
        y = args[-1]
        preds = self.predict(*args[:-1])
        mae = mean_absolute_error(y, preds, multioutput='raw_values')
        cvmae = np.array(mae / y.mean())
        return mae, cvmae

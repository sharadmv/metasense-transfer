from abc import ABCMeta, abstractmethod
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class Model(object, metaclass=ABCMeta):

    def __init__(self, features):
        self.features = features

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def score(self, *args):
        y = args[-1]
        preds = self.predict(*args[:-1])
        y = y.as_matrix()
        mean_y = y.mean(axis=0)
        mean_pred = preds.mean(axis=0)
        crmse = np.sqrt(np.square((preds - mean_pred[None]) - (y - mean_y[None])).mean(axis=0))
        mae = mean_absolute_error(y, preds, multioutput='raw_values')
        mse = mean_squared_error(y, preds, multioutput='raw_values')
        r2 = r2_score(y, preds, multioutput='raw_values')
        rmse = np.sqrt(mse)
        mbe = (y - preds).mean(axis=0)
        cvmae = np.array(mae / y.mean(axis=0))
        return {
            'MAE': mae,
            'CvMAE': cvmae,
            'MBE': mbe,
            'MSE': mse,
            'rMSE': rmse,
            'R^2': r2,
            'crMSE': crmse
        }, preds

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
        return self._score(y, preds), preds

    def _score(self, y_true, y_pred):
        mean_y = y_true.mean(axis=0)
        mean_pred = y_pred.mean(axis=0)
        crmse = np.sqrt(np.square((y_pred - mean_pred[None]) - (y_true - mean_y[None])).mean(axis=0))
        mae = mean_absolute_error(y_true, y_pred, multioutput='raw_values')
        mse = mean_squared_error(y_true, y_pred, multioutput='raw_values')
        r2 = r2_score(y_true, y_pred, multioutput='raw_values')
        rmse = np.sqrt(mse)
        mbe = (y_true - y_pred).mean(axis=0)
        cvmae = np.array(mae / y_true.mean(axis=0))
        return {
            'MAE': mae,
            'CvMAE': cvmae,
            'MBE': mbe,
            'MSE': mse,
            'rMSE': rmse,
            'R^2': r2,
            'crMSE': crmse
        }

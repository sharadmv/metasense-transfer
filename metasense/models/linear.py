import numpy as np
from sklearn.linear_model import LinearRegression

from .model import Model

class Linear(Model):

    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

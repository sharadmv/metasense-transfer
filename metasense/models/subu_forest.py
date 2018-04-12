import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error

from .model import Model

class SubuForest(Model):

    def __init__(self, features):
        super(SubuForest, self).__init__(features)
        self.FEATURES = {2, 4, len(features)}
        self.folds = KFold(5)
        self.models = []

    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        for i, (train_idx, test_idx) in enumerate(self.folds.split(X)):
            # print("Fold #%u" % (i + 1))
            # print("=========================================")
            X_train, y_train = X[train_idx], y[train_idx]
            best = (float('inf'), None)
            X_test, y_test = X[test_idx], y[test_idx]
            for num_features in self.FEATURES:
                cf = MultiOutputRegressor(RandomForestRegressor(max_features=num_features, n_estimators=100, n_jobs=-1))
                cf.fit(X_train, y_train)
                y_pred = cf.predict(X_test)
                error = mean_absolute_error(y_test, y_pred)
                if error < best[0]:
                    best = (error, cf)
            self.models.append(best[1])
        return self

    def predict(self, X):
        preds = np.stack([model.predict(X) for model in self.models])
        return preds.mean(axis=0)

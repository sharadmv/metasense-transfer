from deepx import T

class SplitModel(object):

    def __init__(self, sensor_model, environment_model):
        self.sensor_model, self.environment_model = sensor_model, environment_model

        self.X = T.placeholder([None, 3])
        self.env = T.placeholder([None, 3])
        self.reps = self.sensor_model(self.X)
        self.Y_pred = self.environment_model(self.reps)
        self.Y = T.placeholder([None, 3])

    def fit(self, X, y):
        sess = T.get_current_session()
        self.model.fit(X, y)

    def predict(self, X):
        rep = self.model.predict(X)

from deepx import T
from sklearn.model_selection import train_test_split
import tensorflow as tf

class SensorModel(object):

    def __init__(self, model):
        self.model = model
        self.X = T.placeholder([None, 3])
        self.Y_pred = self.model(self.X)

    def forward(self, sensor_readings):
        return self.model(sensor_readings)

    def predict(self, sensor_readings):
        sess = T.get_current_session()
        return sess.run(self.Y_pred, {
            self.X : sensor_readings
        })

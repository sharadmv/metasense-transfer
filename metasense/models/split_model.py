import random
import joblib
import pickle
import tqdm
import numpy as np
from deepx import T
from sklearn.model_selection import train_test_split

from .model import Model

class SplitModel(Model):

    def __init__(self, sensor_models, calibration_model, lr=1e-4, batch_size=20, log_dir=None, **kwargs):
        self.graph = T.core.Graph()
        self.log_dir = log_dir
        with self.graph.as_default():
            self.calibration_model = calibration_model
            self.board_ids = list(sensor_models.keys())
            self.board_map = {b:i for i, b in enumerate(self.board_ids)}
            self.sensor_map = sensor_models
            self.sensor_models = [sensor_models[board_id] for board_id in self.board_ids]
            self.architecture = pickle.dumps([sensor_models, calibration_model])
            self.batch_size = batch_size
            self.lr = lr

            self.learning_rate = T.placeholder(T.floatx(), [])
            self.sensors = T.placeholder(T.floatx(), [None, 3])
            self.env = T.placeholder(T.floatx(), [None, 3])
            self.board = T.placeholder(T.core.int32, [None])
            self.boards = T.transpose(T.pack([self.board, T.range(T.shape(self.board)[0])]))
            self.rep = T.gather_nd(T.pack([sensor_model(self.sensors) for sensor_model in self.sensor_models]), self.boards)
            self.rep_ = T.placeholder(T.floatx(), [None, self.rep.get_shape()[-1]])
            rep_env = T.concat([self.rep, self.env], -1)
            rep_env_ = T.concat([self.rep_, self.env], -1)
            self.y_ = self.calibration_model(rep_env)
            self.y_rep = self.calibration_model(rep_env_)
            self.y = T.placeholder(T.floatx(), [None, 2])
            self.loss = T.mean((self.y - self.y_) ** 2)
            self.mae = T.mean(T.abs(self.y - self.y_))
            T.core.summary.scalar('MSE', self.loss)
            T.core.summary.scalar('MAE', self.mae)
            self.summary = T.core.summary.merge_all()
            self.train_op = T.core.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        self.session = T.interactive_session(graph=self.graph)

    def fit(self, sensor, env, board, y, n_iters=2000000, seed=0, dump_every=None, cb=None):
        sensor, env, board, y = np.array(sensor), np.array(env), np.array(board), np.array(y)
        sensor_train, sensor_valid, env_train, env_valid, board_train, board_valid, y_train, y_valid = train_test_split(sensor, env, board, y, test_size=0.2, random_state=seed)
        N = sensor_train.shape[0]
        best = (None, float('inf'))
        if self.log_dir is not None:
            writer = T.core.summary.FileWriter(str(self.log_dir), self.graph)
        else:
            writer = None
        for i in tqdm.trange(n_iters):
            idx = random.sample(range(N), self.batch_size)
            if writer is None:
                _, loss = self.session.run([self.train_op, self.loss], {
                    self.sensors: sensor_train[idx],
                    self.env: env_train[idx],
                    self.board: [self.board_map[b] for b in board_train[idx]],
                    self.y: y_train[idx],
                    self.learning_rate: self.lr / 100 if i > int(0.8 * n_iters) else self.lr
                })
            else:
                _, loss, summary = self.session.run([self.train_op, self.loss, self.summary], {
                    self.sensors: sensor_train[idx],
                    self.env: env_train[idx],
                    self.board: [self.board_map[b] for b in board_train[idx]],
                    self.y: y_train[idx],
                    self.learning_rate: self.lr / 100 if i > int(0.8 * n_iters) else self.lr
                })
                writer.add_summary(summary, i)
            if i % 10000 == 0:
                score, _ = self.score(sensor_valid, env_valid, board_valid, y_valid)
                if score['MAE'].mean() < best[1]:
                    best = (
                        self.get_weights(), score['MAE'].mean()
                    )
                    print("New Best:", best[1], score['MAE'])
            if i % 10000 == 0 and dump_every is not None:
                cb(self)
        score, _ = self.score(sensor_valid, env_valid, board_valid, y_valid)
        if score['MAE'].mean() < best[1]:
            best = (
                self.get_weights(), score['MAE'].mean()
            )
            print("New Best:", best[1], score['MAE'])
        self.set_weights(best[0])
        return self

    def get_weights(self):
        return self.session.run([{
                b: model.get_parameters() for b, model in self.sensor_map.items()
        }, self.calibration_model.get_parameters()])

    def set_weights(self, weights):
        sensor_weights, calibration_weights = weights
        self.session.run([
            T.core.assign(a, b) for a, b in zip(self.calibration_model.get_parameters(), calibration_weights)
        ])
        self.session.run([
            T.core.assign(a, b) for sensor in sensor_weights.keys()
            for a, b in zip(self.sensor_map[sensor].get_parameters(), sensor_weights[sensor])
        ])

    def representation(self, sensor, board):
        return self.session.run(self.rep, {
            self.sensors: sensor,
            self.board: [self.board_map[b] for b in board],
        })

    def calibrate(self, rep, env):
        return self.session.run(self.y_rep, {
            self.rep_: rep,
            self.env: env
        })


    def predict(self, sensor, env, board):
        return self.session.run(self.y_, {
            self.sensors: sensor,
            self.env: env,
            self.board: [self.board_map[b] for b in board],
        })

    def __getstate__(self):
        return dict(
            architecture=self.architecture,
            params=self.get_weights(),
            batch_size=self.batch_size,
            lr=self.lr
        )

    def __setstate__(self, state):
        sensor_models, calibration_model = pickle.loads(state['architecture'])
        weights = state['params']
        self.__init__(sensor_models, calibration_model, lr=state['lr'], batch_size=state['batch_size'])
        self.set_weights(weights)

import pickle
import tqdm
import numpy as np
from deepx.nn import *
from deepx import T
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from .model import Model

class NeuralNetwork(Model):

    def __init__(self, model=None, batch_size=20, lr=1e-4):
        self.graph = T.core.Graph()
        with self.graph.as_default():
            self.architecture = pickle.dumps(model)
            self.model = model #Relu(6, 200) >> Relu(200) >> Relu(200) >> Relu(200) >> Linear(2)
            self.batch_size = batch_size
            self.lr = lr

            self.X = T.placeholder(T.floatx(), [None, 6])
            self.y = T.placeholder(T.floatx(), [None, 2])
            self.y_ = self.model(self.X)
            self.loss = T.mean((self.y - self.y_) ** 2)
            self.train_op = T.core.train.AdamOptimizer(self.lr).minimize(self.loss)

        self.session = T.interactive_session(graph=self.graph)


    def fit(self, X, y, n_iters=100000):
        X, y = np.array(X), np.array(y)
        X_train, X_valid, y_train, y_valid= train_test_split(X, y, test_size=0.2)
        N = X_train.shape[0]
        for i in tqdm.trange(n_iters):
            idx = np.random.permutation(N)[:self.batch_size]
            _, loss = self.session.run([self.train_op, self.loss], {
                self.X: X_train[idx],
                self.y: y_train[idx],
            })
            if i % 1000 == 0:
                print(self.score(X_valid, y_valid))
        return self

    def predict(self, X):
        return self.session.run(self.y_, {
            self.X: X
        })

    def __getstate__(self):
        return dict(
            architecture=self.architecture,
            params=self.session.run(self.model.get_parameters()),
            batch_size=self.batch_size,
            lr=self.lr
        )

    def __setstate__(self, state):
        model = pickle.loads(state['architecture'])
        self.__init__(model=model, lr=state['lr'], batch_size=state['batch_size'])
        self.session.run([T.core.assign(a, b) for a, b in zip(self.model.get_parameters(), state['params'])])

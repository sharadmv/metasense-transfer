import random
from deepx import T
import pickle
import tqdm
from argparse import ArgumentParser
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split

from metasense import BOARD_CONFIGURATION as DATA
from metasense.data import load


sensor_features = ['no2', 'o3', 'co']
env_features = ['temperature', 'absolute-humidity', 'pressure']
Y_features = ['epa-no2', 'epa-o3']

def parse_args():
    argparser = ArgumentParser()
    argparser.add_argument('name')
    argparser.add_argument('round1', type=int)
    argparser.add_argument('location1')
    argparser.add_argument('board1', type=int)
    argparser.add_argument('round2', type=int)
    argparser.add_argument('location2')
    argparser.add_argument('board2', type=int)
    argparser.add_argument('--seed', type=int, default=0)
    return argparser.parse_args()

def level1(out_dir):
    RESULTS = {}

def fit_nn(X_data, Y_data, net, batch_size=20, n_iters=100000):

    N = X_data.shape[0]

    for i in tqdm.trange(n_iters):
        idx = random.sample(range(N), batch_size)
        _, l = sess.run([train_op, loss], {
            X: X_data[idx],
            Y: Y_data[idx]
        })
        if i % 1000 == 0:
            print(l)

if __name__ == "__main__":
    args = parse_args()
    model_path = Path('results') / args.name / 'models' / 'model_latest.pkl'
    dataset1 = load(args.round1, args.location1, args.board1)
    dataset2 = load(args.round2, args.location2, args.board2)
    train = dataset1[0].join(dataset2[0], lsuffix='-left').dropna()
    test = dataset1[1].join(dataset2[1], lsuffix='-left').dropna()
    model = joblib.load(model_path)
    fixer_model = pickle.loads(model.architecture)[0][args.board2]

    X = T.placeholder(T.floatx(), [None, 3])
    Y = T.placeholder(T.floatx(), [None, 3])

    Y_ = fixer_model(X)
    loss = T.mean((Y - Y_) ** 2)
    train_op = T.core.train.AdamOptimizer(1e-4).minimize(loss, var_list=fixer_model.get_parameters())

    X_data_train = train[[s + '-left' for s in sensor_features]].as_matrix()
    Y_data_train = model.representation(train[sensor_features], train['board'])
    X_data_test = test[[s + '-left' for s in sensor_features]].as_matrix()
    Y_data_test = model.representation(test[sensor_features], test['board'])

    sess = T.interactive_session()
    fit_nn(X_data_train, Y_data_train, fixer_model, batch_size=64)
    train_preds = model.calibrate(sess.run(Y_, { X: X_data_train }), train[env_features])
    train_mae = abs(train_preds - train[Y_features])
    test_preds = model.calibrate(sess.run(Y_, { X: X_data_test }), test[env_features])
    test_mae = abs(test_preds - test[Y_features])

import random
import s3fs
from deepx import T
import pickle
import tqdm
from argparse import ArgumentParser
import numpy as np
import joblib
from path import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split

from metasense import BOARD_CONFIGURATION
from metasense.data import load

BUCKET_NAME = "metasense-paper-results"

sensor_features = ['no2', 'o3', 'co']
env_features = ['temperature', 'absolute-humidity', 'pressure']
Y_features = ['epa-no2', 'epa-o3']

def parse_args():
    argparser = ArgumentParser()
    argparser.add_argument('experiment')
    argparser.add_argument('out_path')
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
    fs = s3fs.S3FileSystem(anon=False)
    experiment_dir = Path("s3://" + BUCKET_NAME) / args.experiment
    models = [Path(f) for f in fs.ls(experiment_dir)]


    all_triples = set()
    for round in BOARD_CONFIGURATION:
        for location in BOARD_CONFIGURATION[round]:
            for board in BOARD_CONFIGURATION[round][location]:
                all_triples.add((round, location, board))

    results = pd.DataFrame()

    item_map = lambda x: (int(x[0]), x[1], int(x[2]))
    for model in tqdm.tqdm(models):
        # model_triples = set((int(item[0]),item[1], int(item[2])) for name in model.basename().split('-') for item in name.split("_"))
        model_triples = set(item_map(name.split("_")) for name in model.basename().split('-'))
        model_path = model / 'models' / 'model.pkl'
        with fs.open(model_path, 'rb') as fp:
            model = joblib.load(fp)
        for model_triple in model_triples:
            print(model_triple)
            trainable_triples = [triple for triple in (all_triples - model_triples) if model_triple[0] == triple[0] and model_triple[1] == triple[1]]
            dataset1 = load(*model_triple)
            dataset2 = [
                load(*tt) for tt in trainable_triples
            ]
            train = dataset1[0].join(pd.concat([d[0] for d in dataset2]), lsuffix='-left').dropna()
            test = dataset1[1].join(pd.concat([d[1] for d in dataset2]), lsuffix='-left').dropna()
            fixer_model = pickle.loads(model.architecture)[0][trainable_triples[0][2]]

            np.random.seed(args.seed)
            T.core.set_random_seed(args.seed)
            random.seed(args.seed)

            graph = T.core.Graph()
            with graph.as_default():
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
                test_preds = model.calibrate(sess.run(Y_, { X: X_data_test }), test[env_features].as_matrix())
                test_score = model._score(test[Y_features].as_matrix(), test_preds)
                result = {}
                result['Test'] = str(model_triple)
                for i, gas in enumerate(['NO2', 'O3']):
                    for metric, value in test_score.items():
                        result["%s %s" % (gas, metric)] = value[i]
                results = results.append(result, ignore_index=True)
    print(results.describe())
    with open(args.out_path, 'w') as fp:
        fp.write(results.to_csv())

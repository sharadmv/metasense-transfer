import tqdm
import pandas as pd
from argparse import ArgumentParser
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from metasense import BOARD_CONFIGURATION as DATA
from metasense.data import load
from deepx import nn
from metasense.models import SubuForest, Linear, NeuralNetwork

X_features = ['no2', 'o3', 'co', 'temperature', 'humidity', 'pressure']
Y_features = ['epa-no2', 'epa-o3']

def parse_args():
    argparser = ArgumentParser()
    argparser.add_argument('name')
    argparser.add_argument('--level1', action='store_true')
    argparser.add_argument('--level2', action='store_true')
    argparser.add_argument('--level3', action='store_true')
    argparser.add_argument('--model', default='subu')
    argparser.add_argument('--ignore-feature', type=str, action='append', default=[])
    argparser.add_argument('--seed', type=int, default=0)
    return argparser.parse_args()

def level1(out_dir, X_features):
    (out_dir / 'level1' / 'models').mkdir(exist_ok=True, parents=True)
    for round in DATA:
        for location in DATA[round]:
            for board_id in DATA[round][location]:
                print("Training: Round %u - %s - Board %u" % (round, location, board_id))
                train, _ = load(round, location, board_id)
                joblib.dump(
                    (
                        (round, location, board_id),
                        Model(X_features).fit(train[X_features], train[Y_features])
                    ), out_dir / 'level1' / 'models' / ('round%u_%s_board%u.pkl' % (round, location, board_id))
                )

def level2(out_dir, X_features):
    (out_dir / 'level2' / 'models').mkdir(exist_ok=True, parents=True)
    boards = {}
    for round in DATA:
        for location in DATA[round]:
            for board_id in DATA[round][location]:
                if board_id not in boards:
                    boards[board_id] = set()
                boards[board_id].add((round, location))
    for board_id in tqdm.tqdm(boards):
        if len(boards[board_id]) != 3:
            continue
        for test_config in boards[board_id]:
            train_config = boards[board_id] - {test_config}
            data = pd.concat([load(*(t[0], t[1], board_id))[0] for t in train_config])
            # test_data = load(*(test_config[0], test_config[1], board_id))
            joblib.dump(
                (
                    (board_id, train_config),
                    Model(X_features).fit(data[X_features], data[Y_features])
                ), out_dir / 'level2' / 'models' / ('board%u_%s.pkl' % (board_id, '-'.join(map(str, list(train_config)))))
            )

def level3(out_dir, X_features, seed):
    (out_dir / 'level3' / 'models').mkdir(exist_ok=True, parents=True)
    boards = {}
    for round in DATA:
        for location in DATA[round]:
            for board_id in DATA[round][location]:
                if board_id not in boards:
                    boards[board_id] = set()
                boards[board_id].add((round, location))
    for board_id in tqdm.tqdm(boards):
        data = [load(*(t[0], t[1], board_id)) for t in boards[board_id]]
        train_data = pd.concat([t[0] for t in data])
        joblib.dump(
            (
                board_id,
                Model(X_features).fit(train_data[X_features], train_data[Y_features])
            ), out_dir / 'level3' / 'models' / ('board%u.pkl' % board_id)
        )

if __name__ == "__main__":
    args = parse_args()
    out_dir = Path('results') / args.name

    out_dir.mkdir(exist_ok=True, parents=True)
    features = X_features[:]
    for feature in args.ignore_feature:
        features.remove(feature)
    if args.model == 'subu':
        Model = SubuForest
    elif args.model == 'linear':
        Model = Linear
    elif args.model == 'nn-2':
        Model = lambda features, : NeuralNetwork(features, nn.Relu(len(features), 200) >> nn.Relu(200) >> nn.Linear(2))
    elif args.model == 'nn-4':
        Model = lambda features: NeuralNetwork(features, nn.Relu(len(features), 500) >> nn.Relu(500) >> nn.Relu(500) >> nn.Relu(500) >> nn.Linear(2))

    if args.level1:
        level1(out_dir, features)
    if args.level2:
        level2(out_dir, features)
    if args.level3:
        level3(out_dir, features, args.seed)

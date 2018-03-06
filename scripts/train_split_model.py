import pandas as pd
from argparse import ArgumentParser
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from metasense import BOARD_CONFIGURATION as DATA
from metasense.data import load
from metasense.models import SplitModel
from deepx import nn

sensor_features = ['no2', 'o3', 'co']
env_features = ['temperature', 'humidity', 'pressure']
Y_features = ['epa-no2', 'epa-o3']

def parse_args():
    argparser = ArgumentParser()
    argparser.add_argument('--model', default='split')
    argparser.add_argument('--seed', type=int, default=0)
    argparser.add_argument('--dim', type=int, default=5)
    argparser.add_argument('--lr', type=float, default=1e-4)
    return argparser.parse_args()

def train(out_dir, dim, seed):
    (out_dir / 'models').mkdir(exist_ok=True, parents=True)
    boards = {}
    for round in DATA:
        for location in DATA[round]:
            for board_id in DATA[round][location]:
                if board_id not in boards:
                    boards[board_id] = set()
                boards[board_id].add((round, location))
    sensor_models = {
        board_id: nn.Relu(3, 200) >> nn.Relu(200) >> nn.Linear(dim) for board_id in boards
    }
    calibration_model = nn.Relu(dim + 3, 500) >> nn.Relu(500) >> nn.Relu(500) >> nn.Relu(500) >> nn.Linear(2)
    split_model = SplitModel(sensor_models, calibration_model, log_dir=out_dir, lr=args.lr)
    data = []
    for board_id in boards:
        board_train = pd.concat([load(*(t[0], t[1], board_id))[0] for t in boards[board_id]])
        board_train['board'] = board_id
        data.append(board_train)
    max_size = max([d.shape[0] for d in data])
    for d in data:
        if d.shape[0] < max_size:
            d.append(d.sample(max_size - d.shape[0], replace=True))
    data = pd.concat(data)
    split_model.fit(data[sensor_features], data[env_features], data['board'], data[Y_features])
    joblib.dump(split_model, out_dir / 'models' / 'model.pkl')

if __name__ == "__main__":
    args = parse_args()
    out_dir = Path('results') / args.model

    out_dir.mkdir(exist_ok=True, parents=True)

    train(out_dir, args.dim, args.seed)

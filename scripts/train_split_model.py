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
env_features = ['temperature', 'absolute-humidity', 'pressure']
Y_features = ['epa-no2', 'epa-o3']

def parse_args():
    argparser = ArgumentParser()
    argparser.add_argument('name')
    argparser.add_argument('--seed', type=int, default=0)
    argparser.add_argument('--location', default=None, type=str)
    argparser.add_argument('--round', default=None, type=int)
    argparser.add_argument('--board', default=None, type=int)
    argparser.add_argument('--dim', type=int, default=3)
    argparser.add_argument('--batch-size', type=int, default=100)
    argparser.add_argument('--lr', type=float, default=1e-4)
    argparser.add_argument('--load', default=None)
    argparser.add_argument('--num-iters', type=int, default=1000000)
    return argparser.parse_args()

def train(out_dir, dim, seed, load_model=None):
    (out_dir / 'models').mkdir(exist_ok=True, parents=True)
    boards = {}
    for round in DATA:
        for location in DATA[round]:
            for board_id in DATA[round][location]:
                if board_id not in boards:
                    boards[board_id] = set()
                boards[board_id].add((round, location))
    if load_model is None:
        sensor_models = {
            # board_id: nn.Relu(100) >> nn.Relu(100) >> nn.Linear(dim) for board_id in boards
            board_id: nn.Linear(3, dim) for board_id in boards
        }
        calibration_model = nn.Relu(dim + 3, 50) >> nn.Relu(50) >> nn.Linear(2)
        split_model = SplitModel(sensor_models, calibration_model, log_dir=out_dir, lr=args.lr, batch_size=args.batch_size)
    else:
        split_model = joblib.load(load_model)
    data = {}
    print("Filtering round: %s" % args.round)
    print("Filtering location: %s" % args.location)
    for board_id in boards:
        board_train = []
        for round, location in boards[board_id]:
            if (args.round, args.location, args.board) == (round, location, board_id):
                print("Removing: ", round, location, board_id)
                continue
            board_train.append(load(*(round, location, board_id))[0])
        if len(board_train) > 0:
            print("Loaded board[%u]: %u" % (board_id, len(board_train)))
            board_train = pd.concat(board_train)
            board_train['board'] = board_id
            if board_id not in data:
                data[board_id] = []
            data[board_id].append(board_train)
    data = [pd.concat(ds) for ds in data.values()]
    max_size = max([d.shape[0] for d in data])
    for i in range(len(data)):
        d = data[i]
        if d.shape[0] < max_size:
            data[i] = d.append(d.sample(max_size - d.shape[0], replace=True))
    data = pd.concat(data)
    split_model.fit(data[sensor_features], data[env_features], data['board'], data[Y_features], dump_every=(out_dir / 'models' / 'model_latest.pkl', 1000), n_iters=args.num_iters)
    joblib.dump(split_model, out_dir / 'models' / 'model.pkl')

if __name__ == "__main__":
    args = parse_args()
    out_dir = Path('results') / args.name

    out_dir.mkdir(exist_ok=True, parents=True)

    train(out_dir, args.dim, args.seed, load_model=args.load)

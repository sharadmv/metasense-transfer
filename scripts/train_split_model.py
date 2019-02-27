import s3fs
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

BUCKET_NAME = "metasense-paper-results"

def parse_args():
    argparser = ArgumentParser()
    argparser.add_argument('experiment')
    argparser.add_argument('name')
    argparser.add_argument('--seed', type=int, default=0)
    argparser.add_argument('--location', default="", type=str)
    argparser.add_argument('--round', default="", type=str)
    argparser.add_argument('--board', default="", type=str)
    argparser.add_argument('--dim', type=int, default=3)
    argparser.add_argument('--batch-size', type=int, default=10)
    argparser.add_argument('--hidden-size', type=int, default=100)
    argparser.add_argument('--lr', type=float, default=1e-4)
    argparser.add_argument('--load', default=None)
    argparser.add_argument('--num-iters', type=int, default=2000000)
    return argparser.parse_args()

def train(out_dir, dim, seed, load_model=None):
    out_path = out_dir / 'models'
    if not (out_path).exists():
        out_path.mkdir()
    boards = {}
    for round in DATA:
        for location in DATA[round]:
            for board_id in DATA[round][location]:
                if board_id not in boards:
                    boards[board_id] = set()
                boards[board_id].add((round, location))
    if load_model is None:
        sensor_models = {
            board_id: nn.Relu(100) >> nn.Relu(100) >> nn.Linear(dim) for board_id in boards
            # board_id: nn.Linear(3, dim) for board_id in boards
        }
        calibration_model = nn.Relu(dim + 3, args.hidden_size) >> nn.Relu(args.hidden_size) >> nn.Linear(2)
        split_model = SplitModel(sensor_models, calibration_model, log_dir=out_dir, lr=args.lr, batch_size=args.batch_size)
    else:
        split_model = joblib.load(load_model)
    data = {}
    print("Filtering: %s" % ignore)
    for board_id in boards:
        board_train = []
        for round, location in boards[board_id]:
            if (round, location, board_id) in ignore:
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
    def cb(model):
        with open(str(out_path / 'model_latest.pkl'), 'wb') as fp:
            joblib.dump(split_model, fp)
    print("Total data size:", data.shape)
    split_model.fit(data[sensor_features], data[env_features], data['board'], data[Y_features], dump_every=(out_dir / 'models' / 'model_latest.pkl', 1000), n_iters=args.num_iters, cb=cb)
    with open(str(out_path / 'model.pkl'), 'wb') as fp:
        joblib.dump(split_model, fp)

if __name__ == "__main__":
    args = parse_args()
    out_dir = Path('out/') / args.experiment / args.name
    # ignore_round = list(map(int, args.round.split(",")))
    # ignore_location = args.location.split(",")
    # ignore_board = list(map(int, args.board.split(",")))
    # ignore = set(zip(ignore_round, ignore_location, ignore_board))
    ignore = set()
    train(out_dir, args.dim, args.seed, load_model=args.load)

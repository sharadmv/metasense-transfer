import s3fs
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

BUCKET_NAME = "metasense-paper-results"

REVERSE_NAME_MAP = {
    'e': 'elcajon',
    's': 'shafter',
    'd': 'donovan',
}
def parse_args():
    argparser = ArgumentParser()
    argparser.add_argument('experiment')
    argparser.add_argument('name')
    argparser.add_argument('--level1', action='store_true')
    argparser.add_argument('--level2', action='store_true')
    argparser.add_argument('--level3', action='store_true')
    argparser.add_argument('--seed', type=int, default=0)
    return argparser.parse_args()

def benchmark(model, board, test=True):
    if isinstance(board, list):
        data = None
        for t in board:
            board_id = t[-1]
            if test:
                d = load(*t)[1]
            else:
                d = load(*t)[0]
            if data is None:
                data = d
            else:
                data = pd.concat([data, d])
        data['board'] = board_id
    else:
        board_id = board[-1]
        if test:
            data = load(*board)[1]
        else:
            data = load(*board)[0]
        data['board'] = board_id
    scores, _ = model.score(data[sensor_features], data[env_features], data['board'], data[Y_features])
    result = {}
    for i, gas in enumerate(["NO2", "O3"]):
        for score, value in scores.items():
            result["%s %s" % (gas, score)] = value[i]
    return result

def get_triples():
    for round in DATA:
        for location in DATA[round]:
            for board in DATA[round][location]:
                yield (round, location, board)

def level1(out_dir, experiment_dir):

    with fs.open(str(experiment_dir / 'models' / 'model.pkl'), 'rb') as fp:
        model = joblib.load(fp)

    all_experiments = set()
    for round in DATA:
        for location in DATA[round]:
            for board in DATA[round][location]:
                all_experiments.add((round, location, board))
    all_experiments = frozenset(all_experiments)
    def process(x):
        print(x)
        a, b, c = x.split("_")
        return (int(a), REVERSE_NAME_MAP[b], int(c))
    experiment = all_experiments - frozenset(map(process, frozenset(args.name.split("-"))))
    boards = set([x[2] for x in experiment])

    differences = pd.DataFrame(columns=[
        # 'Model', 'NO2 MAE', 'O3 MAE', 'NO2 CvMAE', 'O3 CvMAE'
    ])
    train_results = pd.DataFrame(columns=[
        # 'Model', 'NO2 MAE', 'O3 MAE', 'NO2 CvMAE', 'O3 CvMAE'
    ])
    test_results = pd.DataFrame(columns=[
        # 'Model', 'NO2 MAE', 'O3 MAE', 'NO2 CvMAE', 'O3 CvMAE'
    ])
    print("Experiment", experiment)
    for triple in tqdm.tqdm(list(get_triples())):
        if triple[2] not in boards or triple in experiment:
            print("Skipping", triple)
            continue
        train_result = benchmark(model, triple, test=False)
        test_result  = benchmark(model, triple, test=True)
        difference = {}
        for k, v in train_result.items():
            difference[k] = v - test_result[k]
        train_results = train_results.append({
            'Model': triple,
            **train_result,
        }, ignore_index=True)
        test_results = test_results.append({
            'Model': triple,
            **test_result,
        }, ignore_index=True)
        differences = differences.append({
            'Model': triple,
            **difference
        }, ignore_index=True)
    (out_dir / 'level1').mkdir(exist_ok=True, parents=True)
    with open(str(out_dir / 'level1' / 'train.csv'), 'w') as fp:
        fp.write(train_results.sort_values('Model').to_csv())
    with open(str(out_dir / 'level1' / 'train.tex'), 'w') as fp:
        fp.write(train_results.sort_values('Model').to_latex())
    with open(str(out_dir / 'level1' / 'test.csv'), 'w') as fp:
        fp.write(test_results.sort_values('Model').to_csv())
    with open(str(out_dir / 'level1' / 'test.tex'), 'w') as fp:
        fp.write(test_results.sort_values('Model').to_latex())
    with open(str(out_dir / 'level1' / 'difference.csv'), 'w') as fp:
        fp.write(differences.sort_values('Model').to_csv())
    with open(str(out_dir / 'level1' / 'difference.tex'), 'w') as fp:
        fp.write(differences.sort_values('Model').to_latex())

if __name__ == "__main__":
    args = parse_args()

    fs = s3fs.S3FileSystem(anon=False)
    experiment_dir = Path(BUCKET_NAME) / args.experiment / args.name

    out_dir = Path('results') / args.experiment / args.name

    level1(out_dir, experiment_dir)

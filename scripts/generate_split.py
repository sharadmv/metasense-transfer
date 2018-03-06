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
env_features = ['temperature', 'humidity', 'pressure']
Y_features = ['epa-no2', 'epa-o3']

def parse_args():
    argparser = ArgumentParser()
    argparser.add_argument('--level1', action='store_true')
    argparser.add_argument('--level2', action='store_true')
    argparser.add_argument('--level3', action='store_true')
    argparser.add_argument('--model', default='split')
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
    score = model.score(data[sensor_features], data[env_features], data['board'], data[Y_features])
    return np.stack(score)

def get_triples():
    for round in DATA:
        for location in DATA[round]:
            for board in DATA[round][location]:
                yield (round, location, board)

def level1(out_dir):
    RESULTS = {}

    (out_dir / 'level1').mkdir(exist_ok=True, parents=True)
    model = joblib.load(out_dir / 'models' / 'model.pkl')

    for round, location, board in get_triples():
        if (round, location, board) not in RESULTS:
            RESULTS[(round, location, board)] = []
        for round_ in DATA:
            for location_ in DATA[round]:
                if (round, location) == (round_, location_):
                    continue
                if board in DATA[round_][location_]:
                    RESULTS[(round, location, board)].append((round_, location_))


    differences = pd.DataFrame(columns=[
        'Model', 'NO2 MAE', 'O3 MAE', 'NO2 CvMAE', 'O3 CvMAE'
    ])
    train_results = pd.DataFrame(columns=[
        'Model', 'NO2 MAE', 'O3 MAE', 'NO2 CvMAE', 'O3 CvMAE'
    ])
    test_results = pd.DataFrame(columns=[
        'Model', 'NO2 MAE', 'O3 MAE', 'NO2 CvMAE', 'O3 CvMAE'
    ])
    for triple in tqdm.tqdm(list(get_triples())):
        train_result = benchmark(model, triple, test=False)
        test_result  = benchmark(model, triple, test=True)
        difference = train_result - test_result
        train_results = train_results.append({
            'Model': triple,
            'NO2 MAE': train_result[0, 0],
            'O3 MAE': train_result[0, 1],
            'NO2 CvMAE': train_result[1, 0],
            'O3 CvMAE': train_result[1, 1],
        }, ignore_index=True)
        test_results = test_results.append({
            'Model': triple,
            'NO2 MAE': test_result[0, 0],
            'O3 MAE': test_result[0, 1],
            'NO2 CvMAE': test_result[1, 0],
            'O3 CvMAE': test_result[1, 1],
        }, ignore_index=True)
        differences = differences.append({
            'Model': triple, 'NO2 MAE': difference[0, 0],
            'O3 MAE': difference[0, 1],
            'NO2 CvMAE': difference[1, 0],
            'O3 CvMAE': difference[1, 1],
        }, ignore_index=True)
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
    out_dir = Path('results') / args.model

    if args.level1:
        level1(out_dir)
    if args.level2:
        level2(out_dir)
    if args.level3:
        level3(out_dir, args.seed)

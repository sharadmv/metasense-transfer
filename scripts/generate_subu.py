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


X_features = ['no2', 'o3', 'co', 'temperature', 'humidity', 'pressure']
Y_features = ['epa-no2', 'epa-o3']

def parse_args():
    argparser = ArgumentParser()
    argparser.add_argument('--level1', action='store_true')
    argparser.add_argument('--level2', action='store_true')
    argparser.add_argument('--level3', action='store_true')
    argparser.add_argument('--model', default='subu')
    argparser.add_argument('--seed', type=int, default=0)
    return argparser.parse_args()

def benchmark(model, test):
    if isinstance(test, list):
        data = pd.concat([load(*t) for t in test])
    else:
        data = load(*test)
    score = model.score(data[X_features], data[Y_features])
    return np.stack(score)

def get_triples():
    for round in DATA:
        for location in DATA[round]:
            for board in DATA[round][location]:
                yield (round, location, board)

def level1(out_dir):
    RESULTS = {}

    model_dir = out_dir / 'level1' / 'models'

    for round, location, board in get_triples():
        if (round, location, board) not in RESULTS:
            RESULTS[(round, location, board)] = []
        for round_ in DATA:
            for location_ in DATA[round]:
                if (round, location) == (round_, location_):
                    continue
                if board in DATA[round_][location_]:
                    RESULTS[(round, location, board)].append((round_, location_))

    results = pd.DataFrame(columns=[
        'Model', 'NO2 MAE', 'O3 MAE', 'NO2 CvMAE', 'O3 CvMAE'
    ])
    baselines = pd.DataFrame(columns=[
        'Model', 'NO2 MAE', 'O3 MAE', 'NO2 CvMAE', 'O3 CvMAE'
    ])
    for file in tqdm.tqdm(model_dir.glob('*')):
        triple, model = joblib.load(file)
        tests = RESULTS[triple]
        baseline = benchmark(model, triple)
        if len(tests) > 0:
            result  = baseline - np.stack([benchmark(model, test + (triple[-1],)) for test in tests]).mean(axis=0)
            results = results.append({
                'Model': triple,
                'NO2 MAE': result[0, 0],
                'O3 MAE': result[0, 1],
                'NO2 CvMAE': result[1, 0],
                'O3 CvMAE': result[1, 1],
            }, ignore_index=True)
            baselines = baselines.append({
                'Model': triple,
                'NO2 MAE': baseline[0, 0],
                'O3 MAE': baseline[0, 1],
                'NO2 CvMAE': baseline[1, 0],
                'O3 CvMAE': baseline[1, 1],
            }, ignore_index=True)

    with open(str(out_dir / 'level1' / 'results.csv'), 'w') as fp:
        fp.write(results.to_csv())
    with open(str(out_dir / 'level1' / 'results.tex'), 'w') as fp:
        fp.write(results.to_latex())
    with open(str(out_dir / 'level1' / 'baseline.csv'), 'w') as fp:
        fp.write(baselines.to_csv())
    with open(str(out_dir / 'level1' / 'baseline.tex'), 'w') as fp:
        fp.write(baselines.to_latex())

def level2(out_dir):

    model_dir = out_dir / 'level2' / 'models'

    boards = {}
    for round in DATA:
        for location in DATA[round]:
            for board_id in DATA[round][location]:
                if board_id not in boards:
                    boards[board_id] = set()
                boards[board_id].add((round, location))

    results = pd.DataFrame(columns=[
        'Model', 'NO2 MAE', 'O3 MAE', 'NO2 CvMAE', 'O3 CvMAE'
    ])
    baselines = pd.DataFrame(columns=[
        'Model', 'NO2 MAE', 'O3 MAE', 'NO2 CvMAE', 'O3 CvMAE'
    ])
    for model_file in tqdm.tqdm(model_dir.glob('*')):
        (board_id, train_config), model = joblib.load(model_file)
        test_config = list(boards[board_id] - train_config)[0]
        baseline = benchmark(model, [t + (board_id,) for t in train_config])
        result = baseline - benchmark(model, test_config + (board_id,))
        results = results.append({
            'Model': (board_id, train_config),
            'NO2 MAE': result[0, 0],
            'O3 MAE': result[0, 1],
            'NO2 CvMAE': result[1, 0],
            'O3 CvMAE': result[1, 1],
        }, ignore_index=True)
        baselines = baselines.append({
            'Model': (board_id, train_config),
            'NO2 MAE': baseline[0, 0],
            'O3 MAE': baseline[0, 1],
            'NO2 CvMAE': baseline[1, 0],
            'O3 CvMAE': baseline[1, 1],
        }, ignore_index=True)
    with open(str(out_dir / 'level2' / 'results.csv'), 'w') as fp:
        fp.write(results.to_csv())
    with open(str(out_dir / 'level2' / 'results.tex'), 'w') as fp:
        fp.write(results.to_latex())
    with open(str(out_dir / 'level2' / 'baseline.csv'), 'w') as fp:
        fp.write(baselines.to_csv())
    with open(str(out_dir / 'level2' / 'baseline.tex'), 'w') as fp:
        fp.write(baselines.to_latex())

def level3(out_dir, seed):
    model_dir = out_dir / 'level3' / 'models'

    boards = {}
    for round in DATA:
        for location in DATA[round]:
            for board_id in DATA[round][location]:
                if board_id not in boards:
                    boards[board_id] = set()
                boards[board_id].add((round, location))

    results = pd.DataFrame(columns=[
        'Model', 'NO2 MAE', 'O3 MAE', 'NO2 CvMAE', 'O3 CvMAE'
    ])
    baselines = pd.DataFrame(columns=[
        'Model', 'NO2 MAE', 'O3 MAE', 'NO2 CvMAE', 'O3 CvMAE'
    ])
    for model_file in tqdm.tqdm(model_dir.glob('*')):
        board_id, model = joblib.load(model_file)
        data = pd.concat([load(*(t[0], t[1], board_id)) for t in boards[board_id]])
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=seed)
        baseline = np.stack(model.score(train_data[X_features], train_data[Y_features]))
        result = baseline - np.stack(model.score(test_data[X_features], test_data[Y_features]))
        results = results.append({
            'Model': board_id,
            'NO2 MAE': result[0, 0],
            'O3 MAE': result[0, 1],
            'NO2 CvMAE': result[1, 0],
            'O3 CvMAE': result[1, 1],
        }, ignore_index=True)
        baselines = baselines.append({
            'Model': board_id,
            'NO2 MAE': baseline[0, 0],
            'O3 MAE': baseline[0, 1],
            'NO2 CvMAE': baseline[1, 0],
            'O3 CvMAE': baseline[1, 1],
        }, ignore_index=True)
    with open(str(out_dir / 'level3' / 'results.csv'), 'w') as fp:
        fp.write(results.to_csv())
    with open(str(out_dir / 'level3' / 'results.tex'), 'w') as fp:
        fp.write(results.to_latex())
    with open(str(out_dir / 'level3' / 'baseline.csv'), 'w') as fp:
        fp.write(baselines.to_csv())
    with open(str(out_dir / 'level3' / 'baseline.tex'), 'w') as fp:
        fp.write(baselines.to_latex())

if __name__ == "__main__":
    args = parse_args()
    out_dir = Path('results') / args.model

    if args.level1:
        level1(out_dir)
    if args.level2:
        level2(out_dir)
    if args.level3:
        level3(out_dir, args.seed)

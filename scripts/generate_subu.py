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
        data = pd.concat([load(*t)[1] for t in test])
    else:
        data = load(*test)[1]
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


    differences = pd.DataFrame(columns=[
        'Model', 'NO2 MAE', 'O3 MAE', 'NO2 CvMAE', 'O3 CvMAE'
    ])
    train_results = pd.DataFrame(columns=[
        'Model', 'NO2 MAE', 'O3 MAE', 'NO2 CvMAE', 'O3 CvMAE'
    ])
    test_results = pd.DataFrame(columns=[
        'Model', 'NO2 MAE', 'O3 MAE', 'NO2 CvMAE', 'O3 CvMAE'
    ])
    for file in tqdm.tqdm(list(model_dir.glob('*'))):
        triple, model = joblib.load(file)
        tests = RESULTS[triple]
        train_result = benchmark(model, triple)
        if len(tests) > 0:
            test_result  = np.stack([benchmark(model, test + (triple[-1],)) for test in tests]).mean(axis=0)
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
                'Model': triple,
                'NO2 MAE': difference[0, 0],
                'O3 MAE': difference[0, 1],
                'NO2 CvMAE': difference[1, 0],
                'O3 CvMAE': difference[1, 1],
            }, ignore_index=True)
    with open(str(out_dir / 'level1' / 'train.csv'), 'w') as fp:
        fp.write(train_results.to_csv())
    with open(str(out_dir / 'level1' / 'train.tex'), 'w') as fp:
        fp.write(train_results.to_latex())
    with open(str(out_dir / 'level1' / 'test.csv'), 'w') as fp:
        fp.write(test_results.to_csv())
    with open(str(out_dir / 'level1' / 'test.tex'), 'w') as fp:
        fp.write(test_results.to_latex())
    with open(str(out_dir / 'level1' / 'difference.csv'), 'w') as fp:
        fp.write(differences.to_csv())
    with open(str(out_dir / 'level1' / 'difference.tex'), 'w') as fp:
        fp.write(differences.to_latex())

def level2(out_dir):

    model_dir = out_dir / 'level2' / 'models'

    boards = {}
    for round in DATA:
        for location in DATA[round]:
            for board_id in DATA[round][location]:
                if board_id not in boards:
                    boards[board_id] = set()
                boards[board_id].add((round, location))

    differences = pd.DataFrame(columns=[
        'Model', 'NO2 MAE', 'O3 MAE', 'NO2 CvMAE', 'O3 CvMAE'
    ])
    train_results = pd.DataFrame(columns=[
        'Model', 'NO2 MAE', 'O3 MAE', 'NO2 CvMAE', 'O3 CvMAE'
    ])
    test_results = pd.DataFrame(columns=[
        'Model', 'NO2 MAE', 'O3 MAE', 'NO2 CvMAE', 'O3 CvMAE'
    ])
    for model_file in tqdm.tqdm(list(model_dir.glob('*'))):
        (board_id, train_config), model = joblib.load(model_file)
        test_config = list(boards[board_id] - train_config)[0]
        train_result = benchmark(model, [t + (board_id,) for t in train_config])
        test_result = benchmark(model, test_config + (board_id,))
        difference = train_result - test_result
        train_results = train_results.append({
            'Model': (board_id, train_config),
            'NO2 MAE': train_result[0, 0],
            'O3 MAE': train_result[0, 1],
            'NO2 CvMAE': train_result[1, 0],
            'O3 CvMAE': train_result[1, 1],
        }, ignore_index=True)
        test_results = test_results.append({
            'Model': (board_id, train_config),
            'NO2 MAE': test_result[0, 0],
            'O3 MAE': test_result[0, 1],
            'NO2 CvMAE': test_result[1, 0],
            'O3 CvMAE': test_result[1, 1],
        }, ignore_index=True)
        differences = differences.append({
            'Model': (board_id, train_config),
            'NO2 MAE': difference[0, 0],
            'O3 MAE': difference[0, 1],
            'NO2 CvMAE': difference[1, 0],
            'O3 CvMAE': difference[1, 1],
        }, ignore_index=True)
    with open(str(out_dir / 'level2' / 'train.csv'), 'w') as fp:
        fp.write(train_results.to_csv())
    with open(str(out_dir / 'level2' / 'train.tex'), 'w') as fp:
        fp.write(train_results.to_latex())
    with open(str(out_dir / 'level2' / 'test.csv'), 'w') as fp:
        fp.write(test_results.to_csv())
    with open(str(out_dir / 'level2' / 'test.tex'), 'w') as fp:
        fp.write(test_results.to_latex())
    with open(str(out_dir / 'level2' / 'difference.csv'), 'w') as fp:
        fp.write(differences.to_csv())
    with open(str(out_dir / 'level2' / 'difference.tex'), 'w') as fp:
        fp.write(differences.to_latex())

def level3(out_dir, seed):
    model_dir = out_dir / 'level3' / 'models'

    boards = {}
    for round in DATA:
        for location in DATA[round]:
            for board_id in DATA[round][location]:
                if board_id not in boards:
                    boards[board_id] = set()
                boards[board_id].add((round, location))

    differences = pd.DataFrame(columns=[
        'Model', 'NO2 MAE', 'O3 MAE', 'NO2 CvMAE', 'O3 CvMAE'
    ])
    train_results = pd.DataFrame(columns=[
        'Model', 'NO2 MAE', 'O3 MAE', 'NO2 CvMAE', 'O3 CvMAE'
    ])
    test_results = pd.DataFrame(columns=[
        'Model', 'NO2 MAE', 'O3 MAE', 'NO2 CvMAE', 'O3 CvMAE'
    ])
    for model_file in tqdm.tqdm(list(model_dir.glob('*'))):
        board_id, model = joblib.load(model_file)
        data = [load(*(t[0], t[1], board_id)) for t in boards[board_id]]
        train_data = pd.concat([t[0] for t in data])
        test_data = pd.concat([t[1] for t in data])
        train_result = np.stack(model.score(train_data[X_features], train_data[Y_features]))
        test_result = np.stack(model.score(test_data[X_features], test_data[Y_features]))
        difference = train_result - test_result
        train_results = train_results.append({
            'Model': board_id,
            'NO2 MAE': train_result[0, 0],
            'O3 MAE': train_result[0, 1],
            'NO2 CvMAE': train_result[1, 0],
            'O3 CvMAE': train_result[1, 1],
        }, ignore_index=True)
        test_results = test_results.append({
            'Model': board_id,
            'NO2 MAE': test_result[0, 0],
            'O3 MAE': test_result[0, 1],
            'NO2 CvMAE': test_result[1, 0],
            'O3 CvMAE': test_result[1, 1],
        }, ignore_index=True)
        differences = differences.append({
            'Model': board_id,
            'NO2 MAE': difference[0, 0],
            'O3 MAE': difference[0, 1],
            'NO2 CvMAE': difference[1, 0],
            'O3 CvMAE': difference[1, 1],
        }, ignore_index=True)
    with open(str(out_dir / 'level3' / 'train.csv'), 'w') as fp:
        fp.write(train_results.to_csv())
    with open(str(out_dir / 'level3' / 'train.tex'), 'w') as fp:
        fp.write(train_results.to_latex())
    with open(str(out_dir / 'level3' / 'test.csv'), 'w') as fp:
        fp.write(test_results.to_csv())
    with open(str(out_dir / 'level3' / 'test.tex'), 'w') as fp:
        fp.write(test_results.to_latex())
    with open(str(out_dir / 'level3' / 'difference.csv'), 'w') as fp:
        fp.write(differences.to_csv())
    with open(str(out_dir / 'level3' / 'difference.tex'), 'w') as fp:
        fp.write(differences.to_latex())

if __name__ == "__main__":
    args = parse_args()
    out_dir = Path('results') / args.model

    if args.level1:
        level1(out_dir)
    if args.level2:
        level2(out_dir)
    if args.level3:
        level3(out_dir, args.seed)

import os
import s3fs
import tqdm
from argparse import ArgumentParser
import numpy as np
import joblib
from path import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import OrderedDict

from metasense import BOARD_CONFIGURATION as DATA
from metasense.data import load


X_features = ['no2', 'o3', 'co', 'temperature', 'absolute-humidity', 'pressure']
Y_features = ['epa-no2', 'epa-o3']

BUCKET_NAME = "metasense-paper-results"

def parse_args():
    argparser = ArgumentParser()
    argparser.add_argument('experiment')
    argparser.add_argument('name')
    argparser.add_argument('--level0', action='store_true')
    argparser.add_argument('--level1', action='store_true')
    argparser.add_argument('--level2', action='store_true')
    argparser.add_argument('--level25', action='store_true')
    argparser.add_argument('--level3', action='store_true')
    argparser.add_argument('--level4', action='store_true')
    argparser.add_argument('--seed', type=int, default=0)
    return argparser.parse_args()

def benchmark(model, test, train=False, dump_preds=None):
    idx = 0 if train else 1
    if isinstance(test, list):
        data = pd.concat([load(*t)[idx] for t in test])
    else:
        data = load(*test)[idx]
    if hasattr(model, 'features') and model.features is not None:
        features = model.features
    else:
        features = X_features
    scores, preds = model.score(data[features], data[Y_features])
    if dump_preds is not None:
        dump_preds = Path(dump_preds)
        if not dump_preds.parent.exists():
            os.makedirs(dump_preds.parent, exist_ok=True)
        with open(dump_preds, 'w') as fp:
            df = data.copy()
            df['preds-no2'] = preds[:, 0]
            df['preds-o3'] = preds[:, 1]
            fp.write(df.to_csv())
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


def level0(out_dir, experiment_dir):

    model_dir = experiment_dir / 'level1' / 'models'
    (out_dir / 'level0').mkdir_p()

    differences = pd.DataFrame(columns=[
        'Model', 'Testing Location', 'NO2 MAE', 'O3 MAE', 'NO2 CvMAE', 'O3 CvMAE'
    ])
    train_results = pd.DataFrame(columns=[
        'Model', 'Testing Location', 'NO2 MAE', 'O3 MAE', 'NO2 CvMAE', 'O3 CvMAE'
    ])
    test_results = pd.DataFrame(columns=[
        'Model', 'Testing Location', 'NO2 MAE', 'O3 MAE', 'NO2 CvMAE', 'O3 CvMAE'
    ])
    for file in tqdm.tqdm(list(fs.glob(str(model_dir / '*')))):
        with fs.open(file, 'rb') as fp:
            triple, model = joblib.load(fp)
        train_result = benchmark(model, triple, train=True, dump_preds=DATA_DIR / 'level0/round{round}/{location}/board{board}_train.csv'.format(
            round=triple[0],
            location=triple[1],
            board=triple[2],
        ))
        test_result = benchmark(model, triple, train=False, dump_preds=DATA_DIR / 'level0/round{round}/{location}/board{board}_test.csv'.format(
            round=triple[0],
            location=triple[1],
            board=triple[2],
        ))
        difference = {}
        for k, v in train_result.items():
            difference[k] = v - test_result[k]
        train_results = train_results.append({
            'Model': triple,
            'Testing Location': triple,
            **train_result
        }, ignore_index=True)
        test_results = test_results.append(({
            'Model': triple,
            'Testing Location': triple,
            **test_result
        }), ignore_index=True)
        differences = differences.append(({
            'Model': triple,
            'Testing Location': triple,
            **difference
        }), ignore_index=True)
    with open(str(out_dir / 'level0' / 'train.csv'), 'w') as fp:
        fp.write(train_results.sort_values(['Model', 'Testing Location']).to_csv())
    with open(str(out_dir / 'level0' / 'train.tex'), 'w') as fp:
        fp.write(train_results.sort_values(['Model', 'Testing Location']).to_latex())
    with open(str(out_dir / 'level0' / 'test.csv'), 'w') as fp:
        fp.write(test_results.sort_values(['Model', 'Testing Location']).to_csv())
    with open(str(out_dir / 'level0' / 'test.tex'), 'w') as fp:
        fp.write(test_results.sort_values(['Model', 'Testing Location']).to_latex())
    with open(str(out_dir / 'level0' / 'difference.csv'), 'w') as fp:
        fp.write(differences.sort_values(['Model', 'Testing Location']).to_csv())
    with open(str(out_dir / 'level0' / 'difference.tex'), 'w') as fp:
        fp.write(differences.sort_values(['Model', 'Testing Location']).to_latex())


def level1(out_dir, experiment_dir):
    RESULTS = {}

    model_dir = experiment_dir / 'level1' / 'models'
    (out_dir / 'level1').mkdir_p()

    for round, location, board in get_triples():
        if (round, location, board) not in RESULTS:
            RESULTS[(round, location, board)] = []
        for round_ in DATA:
            for location_ in DATA[round_]:
                if (round, location) == (round_, location_):
                    continue
                if board in DATA[round_][location_]:
                    RESULTS[(round, location, board)].append((round_, location_))


    differences = pd.DataFrame(columns=[
        'Model', 'Testing Location', 'NO2 MAE', 'O3 MAE', 'NO2 CvMAE', 'O3 CvMAE'
    ])
    train_results = pd.DataFrame(columns=[
        'Model', 'Testing Location', 'NO2 MAE', 'O3 MAE', 'NO2 CvMAE', 'O3 CvMAE'
    ])
    test_results = pd.DataFrame(columns=[
        'Model', 'Testing Location', 'NO2 MAE', 'O3 MAE', 'NO2 CvMAE', 'O3 CvMAE'
    ])
    for file in tqdm.tqdm(list(fs.glob(str(model_dir / '*')))):
        with fs.open(file, 'rb') as fp:
            triple, model = joblib.load(fp)
        tests = RESULTS[triple]
        train_result = benchmark(model, triple, train=False, dump_preds=DATA_DIR / 'level1/round{round}/{location}/board{board}_train.csv'.format(
            round=triple[0],
            location=triple[1],
            board=triple[2],
        ))
        if len(tests) > 0:
            test_result = zip(tests, [benchmark(model, test + (triple[-1],),
                                                train=False,
                                                dump_preds=(DATA_DIR / ('level0/round{round}/{location}/board{train_round}-{train_location}-{board}-test.csv').format(
                                                    train_round=triple[0],
                                                    train_location=triple[1],
                                                    round=test[0],
                                                    location=test[1],
                                                    board=triple[2],
                                                ))) for test in tests])
            for test, tr in test_result:
                difference = {}
                for k, v in tr.items():
                    difference[k] = v - tr[k]
                train_results = train_results.append({
                    'Model': triple,
                    'Testing Location': test,
                    **train_result,
                }, ignore_index=True)
                test_results = test_results.append(({
                    'Model': triple,
                    'Testing Location': test,
                    **tr,
                }), ignore_index=True)
                differences = differences.append(({
                    'Model': triple,
                    'Testing Location': test,
                    **difference
                }), ignore_index=True)
    with open(str(out_dir / 'level1' / 'train.csv'), 'w') as fp:
        fp.write(train_results.sort_values(['Model', 'Testing Location']).to_csv())
    with open(str(out_dir / 'level1' / 'train.tex'), 'w') as fp:
        fp.write(train_results.sort_values(['Model', 'Testing Location']).to_latex())
    with open(str(out_dir / 'level1' / 'test.csv'), 'w') as fp:
        fp.write(test_results.sort_values(['Model', 'Testing Location']).to_csv())
    with open(str(out_dir / 'level1' / 'test.tex'), 'w') as fp:
        fp.write(test_results.sort_values(['Model', 'Testing Location']).to_latex())
    with open(str(out_dir / 'level1' / 'difference.csv'), 'w') as fp:
        fp.write(differences.sort_values(['Model', 'Testing Location']).to_csv())
    with open(str(out_dir / 'level1' / 'difference.tex'), 'w') as fp:
        fp.write(differences.sort_values(['Model', 'Testing Location']).to_latex())

def level2(out_dir, experiment_dir):

    model_dir = experiment_dir / 'level2' / 'models'
    (out_dir / 'level2').mkdir_p()

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
    for model_file in tqdm.tqdm(list(fs.glob(str(model_dir / '*')))):
        print("Loading model...")
        with fs.open(model_file, 'rb') as fp:
            (board_id, train_config), model = joblib.load(fp)
        test_config = list(boards[board_id] - train_config)[0]
        print("Benchmarking train...")
        train_result = benchmark(model, [t + (board_id,) for t in train_config], train=False,
                                 dump_preds=DATA_DIR / 'level2/round{round}/{location}/board{board}_train.csv'.format(
            round=test_config[0],
            location=test_config[1],
            board=board_id
        ))
        print("Benchmarking test...")
        test_result = benchmark(model, test_config + (board_id,), train=False,
                                 dump_preds=DATA_DIR / 'level2/round{round}/{location}/board{board}_test.csv'.format(
            round=test_config[0],
            location=test_config[1],
            board=board_id
        ))
        difference = {}
        for k, v in train_result.items():
            difference[k] = v - test_result[k]
        train_results = train_results.append({
            'Model': (board_id, train_config),
            **train_result,
        }, ignore_index=True)
        test_results = test_results.append({
            'Model': (board_id, train_config),
            **test_result,
        }, ignore_index=True)
        differences = differences.append({
            'Model': (board_id, train_config),
            **difference,
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


def level3(out_dir, experiment_dir, seed):
    model_dir = experiment_dir / 'level3' / 'models'
    (out_dir / 'level3').mkdir_p()

    boards = {}
    for round in DATA:
        for location in DATA[round]:
            for board_id in DATA[round][location]:
                if board_id not in boards:
                    boards[board_id] = set()
                boards[board_id].add((round, location))

    differences = pd.DataFrame(columns=[
        'Model', 'Test', 'NO2 MAE', 'O3 MAE', 'NO2 CvMAE', 'O3 CvMAE'
    ])
    train_results = pd.DataFrame(columns=[
        'Model', 'Test', 'NO2 MAE', 'O3 MAE', 'NO2 CvMAE', 'O3 CvMAE'
    ])
    test_results = pd.DataFrame(columns=[
        'Model', 'Test', 'NO2 MAE', 'O3 MAE', 'NO2 CvMAE', 'O3 CvMAE'
    ])
    for model_file in tqdm.tqdm(list(fs.glob(str(model_dir / '*')))):
        with fs.open(model_file, 'rb') as fp:
            board_id, model = joblib.load(fp)
        for t in boards[board_id]:
            train_data, test_data = load(*(t[0], t[1], board_id))
            # train_data = pd.concat([t[0] for t in data])
            # test_data = pd.concat([t[1] for t in data])
            train_result = benchmark(model, (t[0], t[1], board_id), train=True,
                                    dump_preds=DATA_DIR / 'level3/round{round}/{location}/board{board}_train.csv'.format(
                round=t[0],
                location=t[1],
                board=board_id
            ))
            test_result = benchmark(model, (t[0], t[1], board_id), train=False,
                                    dump_preds=DATA_DIR / 'level3/round{round}/{location}/board{board}_test.csv'.format(
                round=t[0],
                location=t[1],
                board=board_id
            ))
            difference = {}
            for k, v in train_result.items():
                difference[k] = v - test_result[k]
            train_results = train_results.append({
                'Model': board_id,
                'Test': (t[0], t[1]),
                **train_result,
            }, ignore_index=True)
            test_results = test_results.append({
                'Model': board_id,
                'Test': (t[0], t[1]),
                **test_result,
            }, ignore_index=True)
            differences = differences.append({
                'Model': board_id,
                'Test': (t[0], t[1]),
                **difference
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
    out_dir = Path('results') / args.experiment/ args.name
    out_dir.mkdir_p()

    fs = s3fs.S3FileSystem(anon=False)
    experiment_dir = Path(BUCKET_NAME) / args.experiment / args.name
    DATA_DIR = Path("/media/sharad/big/data/metasense/sd/predictions/%s/%s" % (args.experiment, args.name))

    if args.level0:
        level0(out_dir, experiment_dir)
    if args.level1:
        level1(out_dir, experiment_dir)
    if args.level2:
        level2(out_dir, experiment_dir)
    if args.level25:
        level25(out_dir, experiment_dir)
    if args.level3:
        level3(out_dir, experiment_dir, args.seed)
    if args.level4:
        level4(out_dir, experiment_dir, args.seed)

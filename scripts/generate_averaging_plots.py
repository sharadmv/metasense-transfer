import tqdm
import joblib
import ipdb
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
from argparse import ArgumentParser

def parse_args():
    argparser = ArgumentParser()
    argparser.add_argument('path')

    return argparser.parse_args()

def load_data(model, path):
    data = pd.read_csv(path)
    return data

MODELS = ['linear', 'nn-2', 'nn-4', 'subu']

def process_level(data, txt):
    data['Experiment'] = txt
    return data

if __name__ == "__main__":
    args = parse_args()
    path = Path(args.path)
    models = []
    for m in tqdm.tqdm((path / 'level1' / 'models').glob('*.pkl')):
        models.append(joblib.load(m))

    boards = {}
    for round in DATA:
        for location in data[round]:
            for board_id in data[round][location]:
                if board_id not in boards:
                    boards[board_id] = set()
                boards[board_id].add((round, location))

    for (round, location, board), model in models:
        if

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
METRICS = ['%s %s' % (a, b) for a in
           ["NO2", "O3"] for b in
           ["MAE", "CvMAE", "MSE", "rMSE", "crMSE", "MBE", "R^2"]
]

def process_level(data, text):
    data['Experiment'] = text
    return data

if __name__ == "__main__":
    args = parse_args()
    path = Path(args.path)
    level0 = process_level(pd.read_csv(path / 'level0' / 'test.csv'), 'Level 0')
    level1 = process_level(pd.read_csv(path / 'level1' / 'test.csv'), 'Level 1')
    level2 = process_level(pd.read_csv(path / 'level2' / 'test.csv'), 'Level 2')
    # level25 = process_level(pd.read_csv(path / 'level2.5' / 'test.csv'), 'Level 2.5')
    level3 = process_level(pd.read_csv(path / 'level3' / 'test.csv'), 'Level 3')
    # level4 = process_level(pd.read_csv(path / 'level4' / 'test.csv'), 'Level 4')
    data = pd.concat([level0, level1, level2, level3])
    for metric in METRICS:
        plt.figure()
        sns.boxplot(data=data, x='Experiment', y=metric)
        plt.savefig(str(path / ('%s.png' % metric)))

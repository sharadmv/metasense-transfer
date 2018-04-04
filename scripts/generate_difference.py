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
    data['Predictor'] = model
    data['Location'] = list(map(lambda x: eval(x)[1], data['Model']))
    return data

MODELS = ['linear', 'nn-2', 'nn-4', 'subu']

if __name__ == "__main__":
    args = parse_args()
    path = Path(args.path)
    all_data = pd.concat([load_data(model, path / model / 'level1' / 'difference.csv') for model in MODELS])

    fig = plt.figure()
    sns.boxplot(data=all_data, x='Location', y='NO2 MAE', hue='Predictor')
    fig.suptitle('NO2 MAE')
    fig.savefig(str(path / 'no2mae_diff.png'), bbox_inches='tight')
    fig = plt.figure()
    sns.boxplot(data=all_data, x='Location', y='O3 MAE', hue='Predictor')
    fig.suptitle('O3 MAE')
    fig.savefig(str(path / 'o3mae_diff.png'), bbox_inches='tight')
    fig = plt.figure()
    sns.boxplot(data=all_data, x='Location', y='NO2 CvMAE', hue='Predictor')
    fig.suptitle('NO2 CvMAE')
    fig.savefig(str(path / 'no2cvmae_diff.png'), bbox_inches='tight')
    fig = plt.figure()
    sns.boxplot(data=all_data, x='Location', y='O3 CvMAE', hue='Predictor')
    fig.suptitle('O3 CvMAE')
    fig.savefig(str(path / 'o3cvmae_diff.png'), bbox_inches='tight')

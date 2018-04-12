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
    argparser.add_argument('--models', nargs='+')
    argparser.add_argument('out')

    return argparser.parse_args()

def load_data(model, path):
    data = pd.read_csv(path)
    data['Predictor'] = model
    return data

MODELS = ['linear', 'nn-2', 'nn-4', 'subu']

if __name__ == "__main__":
    args = parse_args()
    path = Path(args.path)
    out = Path(args.out)
    all_data = pd.concat([load_data(model, path / model / 'level1' / 'test.csv') for model in args.models])

    fig = plt.figure()
    sns.boxplot(data=all_data, x='Predictor', y='NO2 MAE')
    fig.suptitle('NO2 MAE')
    fig.savefig(str(out / 'level1-no2mae_diff.png'), bbox_inches='tight')
    fig = plt.figure()
    sns.boxplot(data=all_data, x='Predictor', y='O3 MAE')
    fig.suptitle('O3 MAE')
    fig.savefig(str(out / 'level1-o3mae_diff.png'), bbox_inches='tight')
    fig = plt.figure()
    sns.boxplot(data=all_data, x='Predictor', y='NO2 CvMAE')
    fig.suptitle('NO2 CvMAE')
    fig.savefig(str(out / 'level1-no2cvmae_diff.png'), bbox_inches='tight')
    fig = plt.figure()
    sns.boxplot(data=all_data, x='Predictor', y='O3 CvMAE')
    fig.suptitle('O3 CvMAE')
    fig.savefig(str(out / 'level1-o3cvmae_diff.png'), bbox_inches='tight')

    all_data = pd.concat([load_data(model, path / model / 'level0' / 'test.csv') for model in args.models])

    fig = plt.figure()
    sns.boxplot(data=all_data, x='Predictor', y='NO2 MAE')
    fig.suptitle('NO2 MAE')
    fig.savefig(str(out / 'level0-no2mae_diff.png'), bbox_inches='tight')
    fig = plt.figure()
    sns.boxplot(data=all_data, x='Predictor', y='O3 MAE')
    fig.suptitle('O3 MAE')
    fig.savefig(str(out / 'level0-o3mae_diff.png'), bbox_inches='tight')
    fig = plt.figure()
    sns.boxplot(data=all_data, x='Predictor', y='NO2 CvMAE')
    fig.suptitle('NO2 CvMAE')
    fig.savefig(str(out / 'level0-no2cvmae_diff.png'), bbox_inches='tight')
    fig = plt.figure()
    sns.boxplot(data=all_data, x='Predictor', y='O3 CvMAE')
    fig.suptitle('O3 CvMAE')
    fig.savefig(str(out / 'level0-o3cvmae_diff.png'), bbox_inches='tight')

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

def load_data(path, type):
    data = pd.read_csv(path)
    data['Location'] = list(map(lambda x: eval(x)[1], data['Model']))
    data['Dataset'] = type
    return data

if __name__ == "__main__":
    args = parse_args()
    path = Path(args.path)
    local_data = load_data(Path(args.path) / 'level0'/ 'test.csv', 'Local Test')
    test_data = load_data(Path(args.path) / 'level1'/ 'test.csv', 'Remote Test')
    train_data = load_data(Path(args.path) / 'level0' / 'train.csv', 'Train')
    all_data = pd.concat([train_data, local_data, test_data])
    fig, ax = plt.subplots(1, 2)
    sns.boxplot(data=all_data, x='Location', y='NO2 MAE', hue='Dataset', ax=ax[0])
    sns.boxplot(data=all_data, x='Location', y='O3 MAE', hue='Dataset', ax=ax[1])
    ax[0].set_title('NO2')
    ax[1].set_title('O3')
    plt.legend(loc='best')
    fig.savefig(str(path / 'level1.png'), bbox_inches='tight')

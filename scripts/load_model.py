import collections
import numpy as np
import tqdm
from metasense.data import load
import click
import joblib
from sklearn.tree import _tree
from path import Path

X_features = ['no2', 'o3', 'co', 'temperature', 'humidity', 'pressure']
Y_features = ['epa-no2', 'epa-o3']

@click.command()
@click.argument('path')
def load_models(path):
    path = Path(path)
    results = []
    for model in tqdm.tqdm(path.glob('*')):
        model = path / 'round2_donovan_board18.pkl'
        tree = joblib.load(model)
        if tree[0] != (2, 'donovan', 18):
            continue
        print(analyze_tree(*tree))
        break

def analyze_tree(config, tree):
    test_data0 = load(2, 'donovan', 18)[0]
    test_data1 = load(3, 'elcajon', 18)[0]
    variances = []
    for i, gas in enumerate(Y_features):
        result = tree.models[0].estimators_[i].apply(test_data0[X_features].as_matrix().astype(np.float32))
        tree_var0 = analyze_tree_result(result, test_data0[gas])
        tree_var1 = analyze_tree_result(result, test_data1[gas])
        variances.append([tree_var0, tree_var1])
    return np.stack(variances)

def analyze_tree_result(result, data):
    i = 0
    sample_set = collections.defaultdict(list)
    for j, s in enumerate(result[:, i]):
        sample_set[s].append(j)
    return np.mean([np.std(data[samples]) for samples in sample_set.values()])

if __name__ == "__main__":
    load_models()

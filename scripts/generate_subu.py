import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from metasense import BOARD_CONFIGURATION as DATA
from metasense.data import load

def get_triples():
    for round in DATA:
        for location in DATA[round]:
            for board in DATA[round][location]:
                yield (round, location, board)

RESULTS = {}

model_dir = Path('results') / 'subu' / 'level1' / 'models'

for round, location, board in get_triples():
    if (round, location, board) not in RESULTS:
        RESULTS[(round, location, board)] = []
    for round_ in DATA:
        for location_ in DATA[round]:
            if (round, location) == (round_, location_):
                continue
            if board in DATA[round_][location_]:
                RESULTS[(round, location, board)].append((round_, location_))

X_features = ['no2', 'o3', 'co', 'temperature', 'humidity', 'pressure']
Y_features = ['epa-no2', 'epa-o3']


out_dir = Path('results') / 'subu'

def benchmark(model, test):
    data = load(*test)
    score = model.score(data[X_features], data[Y_features])
    return np.stack(score)

results = pd.DataFrame(columns=[
    'Model', 'NO2 MAE', 'O3 MAE', 'NO2 CvMAE', 'O3 CvMAE'
])
for file in model_dir.glob('*'):
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

with open(str(out_dir / 'level1' / 'results.csv'), 'w') as fp:
    fp.write(results.to_csv())
with open(str(out_dir / 'level1' / 'results.tex'), 'w') as fp:
    fp.write(results.to_latex())

import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd

def load(round, location, board_id, root_dir=Path('data/final'), seed=0):
    path = root_dir / ("round%u" % round) / location / str("%u.csv" % board_id)
    data = pd.read_csv(path, parse_dates=True, index_col='datetime')
    data['no2'] = data['no2-A'] - data['no2-W']
    data['o3']  = data['o3-A']  - data['o3-W']
    data['co']  = data['co-A']  - data['co-W']
    data['epa-no2'] *= 1000
    data['epa-o3'] *= 1000
    data['epa-no2'] = data['epa-no2'].clip(0, np.inf)
    data['epa-o3'] = data['epa-o3'].clip(0, np.inf)
    T = data['temperature'] + 273.15
    data['absolute-humidity'] = data['humidity'] / 100 * np.exp(
        54.842763 - 6763.22 / T - 4.210 * np.log(T) + 0.000367 * T +
        np.tanh(0.0415 * (T - 218.8)) * (53.878 - 1331.22 / T
                                         - 9.44523 * np.log(T) + 0.014025 * T)) / 1000
    data['board'] = board_id
    data['location'] = location
    data['round'] = round
    data = data[~(data['temperature'] > 65)]
    data = data[~(data['absolute-humidity'] > 10)]
    train, test = train_test_split(data, test_size=0.2, random_state=seed)
    return train, test

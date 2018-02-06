from pathlib import Path
import pandas as pd

def load(round, location, board_id, root_dir=Path('data/final')):
    path = root_dir / ("round%u" % round) / location / str("%u.csv" % board_id)
    data = pd.read_csv(path, parse_dates=True, index_col='datetime')
    data['no2'] = data['no2-A'] - data['no2-W']
    data['o3']  = data['o3-A']  - data['o3-W']
    data['co']  = data['co-A']  - data['co-W']
    data['epa-no2'] *= 1000
    data['epa-o3'] *= 1000
    return data

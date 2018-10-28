import joblib
from pathlib import Path
from metasense import BOARD_CONFIGURATION as DATA
from metasense.data import load
from metasense.models import SubuForest

X_features = ['no2', 'o3', 'co', 'temperature', 'humidity', 'pressure']
Y_features = ['epa-no2', 'epa-o3']


out_dir = Path('results') / 'subu'
(out_dir / 'level1' / 'models').mkdir(exist_ok=True, parents=True)
(out_dir / 'level2' / 'models').mkdir(exist_ok=True, parents=True)

for round in DATA:
    if round not in MODELS:
        MODELS[round] = {}
    for location in DATA[round]:
        if location not in MODELS[round]:
            MODELS[round][location] = {}
        for board_id in DATA[round][location]:
            print("Training: Round %u - %s - Board %u" % (round, location, board_id))
            data = load(round, location, board_id)
            joblib.dump(
                (
                    (round, location, board_id),
                    SubuForest().fit(data[X_features], data[Y_features])
                ), out_dir / 'level1' / 'models' / ('round%u_%s_board%u.pkl' % (round, location, board_id))
            )

import matplotlib.patches as mpatches
import mpl_scatter_density
from mpl_scatter_density import ScatterDensityArtist
import numpy as np
import tqdm
import os
import glob
import matplotlib.pyplot as plt
import pandas as pd
from argparse import ArgumentParser
import seaborn as sns
from path import Path
sns.set(style='white')

def parse_args():
    argparser = ArgumentParser()
    argparser.add_argument('in_path')
    argparser.add_argument('out_path')
    argparser.add_argument('--x_low', type=float)
    argparser.add_argument('--x_high', type=float)
    argparser.add_argument('--y_low', type=float)
    argparser.add_argument('--y_high', type=float)
    return argparser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    in_path, out_path = Path(args.in_path), Path(args.out_path)

    df = pd.DataFrame()
    for file in tqdm.tqdm(glob.glob(in_path / "**/*.csv", recursive=True)):
        print(file)
        data = pd.read_csv(file)
        if "-test" in file:
            name = os.path.basename(file)[5:-4].split("-")[:-1]
            round, location, board = int(name[0]), name[1], int(name[2])
            data["train_round"] = round
            data["train_location"] = location
        else:
            data["train_round"] = data["round"]
            data["train_location"] = data["location"]
        for gas in ["NO2", "O3"]:
            data["%s MAE" % gas] = np.abs(data["epa-%s" % gas.lower()] - data["preds-%s" % gas.lower()])
        df = pd.concat([df, data[["epa-no2", "epa-o3", "train_round", "train_location", "board", "round", "location", "temperature", "absolute-humidity", "pressure", "NO2 MAE", "O3 MAE"]]])

    LOCATIONS = {"elcajon","shafter", "donovan"}
    MAP = {
        "elcajon": "El Cajon",
        "shafter": "Shafter",
        "donovan": "Donovan"
    }
    colors = {
        'donovan': 'red',
        'elcajon': 'green',
        'shafter': 'blue'
    }
    dwf = 75
    for metric in ["epa-no2", "epa-o3", "temperature", "pressure", "absolute-humidity"]:
        for train_location in LOCATIONS:
            for gas in tqdm.tqdm(["NO2", "O3"]):
                fig, ax = plt.subplots(subplot_kw=dict(projection='scatter_density'))
                # ax[0].scatter(filtered[metric], filtered["%s MAE" % gas], label=MAP[train_location], alpha=0.1)
                axes = []
                for i, test_location in enumerate(LOCATIONS - {train_location}):
                    filtered = df[(df['train_location'] == train_location) & (df['location'] == test_location)]
                    error = "%s MAE" % gas
                    # filtered = filtered[np.abs(filtered[metric]-filtered[metric].mean()) <= (2*filtered[metric].std())]
                    # filtered = filtered[np.abs(filtered[error]-filtered[error].mean()) <= (3*filtered[error].std())]
                    axes.append(mpatches.Patch(color=colors[test_location], label=MAP[test_location]))
                    ax.scatter_density(filtered[metric], filtered[error], color=colors[test_location], label=MAP[test_location], dpi=dwf)
                    # sns.kdeplot(filtered[metric], filtered["%s MAE" % gas], ax=ax[i + 1])
                filtered = df[(df['train_location'] == train_location) & (df['location'] == train_location)]
                error = "%s MAE" % gas
                # filtered = filtered[np.abs(filtered[error]-filtered[error].mean()) <= (3*filtered[error].std())]
                ax.scatter_density(filtered[metric], filtered[error], color=colors[train_location], label=MAP[train_location], dpi=dwf)
                axes.append(mpatches.Patch(color=colors[train_location], label=MAP[train_location]))
                ax.legend(handles=axes)
                if gas == "O3" and metric == "absolute-humidity":
                    # ax.set_xlim(args.x_low, args.x_high)
                    ax.set_ylim(args.y_low, args.y_high)
                ax.set_ylabel("%s (ppb)" % error)
                # ax[1].legend(loc='best')
                # ax[2].legend(loc='best')
                fig.savefig(out_path / ("error_density_%s_%s_%s.png" % (metric, train_location, gas)), dpi=200, bbox_inches='tight')
                plt.close(fig)

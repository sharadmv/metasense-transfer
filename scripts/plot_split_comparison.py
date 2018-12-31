import click
import seaborn as sns
from path import Path
import pandas as pd
import matplotlib.pyplot as plt

NAMES = [
    ("results/filtered-data/nn-4/level1", "NN-4"),
    ("results/splits-500-10relu-seasonal-size3/level1", "Split-NN (1)"),
    ("results/splits-500-10relu-seasonal-size6/level1", "Split-NN (2)"),
    ("results/splits-500-10relu-seasonal-size9/level1", "Split-NN (3)"),
    ("results/filtered-data/subu/level1", "RF (1)"),
]


@click.command()
@click.argument('out_path')
@click.argument('type')
def main(out_path, type):
    out_path = Path(out_path)
    data = pd.DataFrame()
    for path, name in NAMES:
        path = Path(path)
        df = pd.read_csv(path / 'test.csv')
        df["Base"] = df["Model"]
        df["Model"] = name
        data = pd.concat([data, df])

    fig, ax = plt.subplots()
    sns.boxplot(
        data=data,
        x='Model',
        y='NO2 MAE',
        showfliers=False,
    )
    fig.savefig(out_path / "split-no2-{type}-mae.png".format(type=type), bbox_inches='tight', dpi=200)

    fig, ax = plt.subplots()
    sns.boxplot(
        data=data,
        x='Model',
        y='O3 MAE',
        showfliers=False,
    )
    fig.savefig(out_path / "split-o3-{type}-mae.png".format(type=type), bbox_inches='tight', dpi=200)
    plt.show()

if __name__ == "__main__":
    main()

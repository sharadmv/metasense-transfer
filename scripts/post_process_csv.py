import pandas as pd
from path import Path
import click
import glob

@click.command()
@click.argument("results_path")
def main(results_path):
    results_path = Path(results_path)
    for csv_path in glob.glob(results_path / "**" / "*.csv", recursive=True):
        csv_path = Path(csv_path)
        csv_basename = csv_path.basename()[:-4]
        csv_parent = csv_path.parent
        data = pd.read_csv(csv_path)
        columns = data.columns
        no2_columns = [c for c in columns if "NO2" in c]
        o3_columns = [c for c in columns if "O3" in c]
        base_columns = set(columns) - set(o3_columns) - set(no2_columns) - {"Unnamed: 0"}
        with open(csv_parent / ("%s-no2.tex" % csv_basename), 'w') as fp:
            fp.write(data[list(base_columns) + no2_columns].to_latex())
        with open(csv_parent / ("%s-o3.tex" % csv_basename), 'w') as fp:
            fp.write(data[list(base_columns) + o3_columns].to_latex())

if __name__ == "__main__":
    main()

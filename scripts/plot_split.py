import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
import pandas as pd
import tqdm
import joblib
from path import Path
import click

METRICS = ["%s %s"% (gas, metric) for gas in ["NO2", "O3"]
           for metric in ["MAE", "CvMAE", "MSE", "rMSE", "crMSE", "R^2", "MBE"]]

def load_subu(subu_path, models, name):
    subu_level1 = pd.read_csv(subu_path / 'level1' / 'test.csv')
    subu_level1['Model'] = subu_level1['Model'].apply(eval)
    subu_level1['Testing Location'] = subu_level1['Testing Location'].apply(eval)
    subu_level1['Board'] = subu_level1['Model'].apply(lambda x: int(x[2]))
    subu_level1['Board'] = subu_level1['Board'].astype(np.int32)
    subu_level1['Model'] = subu_level1[['Board', 'Testing Location']].apply(lambda x: (x[1][0], x[1][1], x[0]), axis=1)
    subu_level1['Benchmark'] = "Level 1"
    subu_level1 = subu_level1.drop("Testing Location", axis=1)
    subu_level2 = pd.read_csv(subu_path / 'level2' / 'test.csv')
    subu_level2['Model'] = subu_level2['Model'].apply(eval)
    subu_level2['Benchmark'] = "Level 2"
    locations = {'elcajon', 'donovan', 'shafter'}
    rounds = {2, 3, 4}
    for i, result in subu_level2.iterrows():
        board_id, trains = result['Model']
        r = set(x[0] for x in trains)
        l = set(x[1] for x in trains)
        triple = (list(rounds - r)[0], list(locations - l)[0], board_id)
        subu_level2.at[i, 'Model'] = triple
        subu_level2.at[i, 'Board'] = board_id
    subu_level2['Board'] = subu_level2['Board'].astype(np.int32)
    subu_level2.loc[:, 'Benchmark'] = "Level 2"
    subu_results = pd.concat([subu_level1, subu_level2])
    subu_results.loc[:, 'Calibration'] = name
    subu_results = subu_results.drop('Unnamed: 0', axis=1)
    return subu_results

def get_results(models):
    results = pd.DataFrame(columns=['Model', 'NO2 MAE', 'O3 MAE', 'NO2 CvMAE', 'O3 CvMAE'])
    for model_name in tqdm.tqdm(models):
        if model_name[-4] == '.' or '_' not in model_name:
            continue
        if len(model_name.split("-")) > 2:
            continue
        model_path = Path(model_name)
        model_split = model_path.split("/")[-1].split("-")
        model_splits = [x.split("_") for x in model_split]
        triples = [(int(x[0]), x[1], int(x[2])) for x in model_splits]
        model_results = pd.read_csv(model_path / 'level1' / 'test.csv')
        model_results = model_results.drop('Unnamed: 0', axis=1)
        model_results['Model'] = model_results['Model'].apply(eval)
        def filter(x):
            return x['Model'] in triples
        filtered = model_results[model_results.apply(filter, axis=1)]
        if len(triples) == 9:
            filtered.loc[:, "Benchmark"] = 'Level 2'
        elif len(triples) == 18:
            filtered.loc[:, "Benchmark"] = 'Level 1'
        else:
            filtered.loc[:, "Benchmark"] = 'Level %u' % (3 - len(triples))
        results = results.append(filtered)
        results.loc[:, 'Calibration'] = 'Split-NN'
    return results, set(results['Model'])

def merge_results(split_results, subu_results, name):
    merged = pd.concat([split_results, subu_results])
    def filter(x):
        for column in METRICS:
            if x['Calibration'] == 'Split-NN':
                x['%s Improvement' % column] = -(x[column] - merged[(merged['Model'] == x['Model']) & (merged['Calibration'] == name) & (merged['Benchmark'] == x['Benchmark'])][column].mean())
            else:
                x['%s Improvement' % column] = np.nan
        return x
    merged = merged.apply(filter, axis=1)
    return merged

def plot_results(results, column, out_path):
    results = results[~results[column].isnull()]
    fig, ax = plt.subplots()
    data = results.copy()
    data["Model"] = data["Calibration"]
    sns.boxplot(hue='Model', y=column, data=data[["Model", "Benchmark"] + METRICS + ["%s Improvement" % x for x in METRICS]], ax=ax, x='Benchmark', whis=2)
    fig.savefig(out_path, bbox_inches='tight')

@click.command()
@click.argument('out_dir', nargs=1)
@click.argument('name', nargs=1)
@click.argument('out_path', nargs=1)
@click.argument('subu_path', nargs=1)
@click.argument('models', nargs=-1, type=str)
def main(name, out_dir, out_path, subu_path, models):
    out_path = Path(out_path)
    if models == ['*']:
        split_results, split_models = get_results([out_path / m for m in models])
    else:
        split_results, split_models = get_results(out_path.parent.listdir())
    out_path = out_path / out_dir
    out_path.mkdir_p()
    subu_results = load_subu(Path(subu_path), split_models, name)
    merged_results = merge_results(split_results, subu_results, name)
    merged_results.to_csv(out_path / 'results.csv')
    merged_results.to_latex(out_path / 'results.tex')

    for metric in METRICS:
        result = merged_results[merged_results['Calibration'] == 'Split-NN'][metric].dropna().describe()
        print("%s[Split]: %.3f ± %.3f" % (metric, result['mean'], result['std']))
        result = merged_results[merged_results['Calibration'] == name][metric].dropna().describe()
        print("%s[%s]: %.3f ± %.3f" % (metric, name, result['mean'], result['std']))

        result = merged_results['%s Improvement' % metric].dropna().describe()
        print("%s Improvement: %.3f ± %.3f" % (metric, result['mean'], result['std']))
        merged_results = merged_results.sort_values('Benchmark')

        plot_results(merged_results, metric, out_path / ('%s.png' % metric))

        plot_results(merged_results, '%s Improvement' % metric, out_path / ('%s-diff.png' % metric))

if __name__ == '__main__':
    main()

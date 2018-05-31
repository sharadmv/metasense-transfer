import glob
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
import pandas as pd
import tqdm
import joblib
from path import Path
import click

def load_subu(subu_path, models):
    subu_level1 = pd.read_csv(subu_path / 'level1' / 'test.csv')
    subu_level1['Model'] = subu_level1['Model'].apply(eval)
    subu_level1['Testing Location'] = subu_level1['Testing Location'].apply(eval)
    subu_level1['Board'] = subu_level1['Model'].apply(lambda x: x[2])
    subu_level1['Model'] = subu_level1[['Board', 'Testing Location']].apply(lambda x: (x[1][0], x[1][1], x[0]), axis=1)
    subu_level1['Training Size'] = 1
    subu_results = pd.read_csv(subu_path / 'level2' / 'test.csv')
    subu_results['Model'] = subu_results['Model'].apply(eval)
    locations = {'elcajon', 'donovan', 'shafter'}
    rounds = {1, 2, 3}
    for i, result in subu_results.iterrows():
        board_id, trains = result['Model']
        r = set(x[0] for x in trains)
        l = set(x[1] for x in trains)
        triple = (list(rounds - r)[0], list(locations - l)[0], board_id)
        subu_results.at[i, 'Model'] = triple
    subu_results['Training Size'] = 2
    subu_models = set(subu_results['Model'])
    for model in models:
        if model not in subu_models:
            level1 = subu_level1[subu_level1['Model'] == model]
            assert len(level1) == 1
            subu_results = subu_results.append(level1)
    subu_results['Calibration'] = 'Random Forest'
    subu_results = subu_results.drop('Unnamed: 0', axis=1)
    return subu_results

def get_results(models):
    results = pd.DataFrame(columns=['Model', 'NO2 MAE', 'O3 MAE', 'NO2 CvMAE', 'O3 CvMAE'])
    for model_name in tqdm.tqdm(models):
        model_path = Path(model_name)
        model_split = model_path.split('/')[-1].split('-')
        triple = (int(model_split[0]), model_split[1], int(model_split[2]))
        model_results = pd.read_csv(model_path / 'level1' / 'test.csv')
        model_results = model_results.drop('Unnamed: 0', axis=1)
        model_results['Model'] = model_results['Model'].apply(eval)
        filtered = model_results[model_results['Model'] == triple]
        results = results.append(filtered)
    results['Calibration'] = 'Split-NN'
    return results, set(results['Model'])

def merge_results(split_results, subu_results):
    merged = pd.DataFrame(columns=split_results.columns)
    split_models, subu_models = set(split_results['Model']), set(subu_results['Model'])
    all_models = split_models | subu_models
    for model in all_models:
        if not model in subu_models:
            print("No subu match:", model)
            continue
        if not model in split_models:
            print("No split match:", model)
            continue
        split_intersection = split_results[split_results['Model'] == model]
        subu_intersection = subu_results[subu_results['Model'] == model]
        assert len(split_intersection) == 1 and len(subu_intersection) == 1
        split_intersection = split_intersection.iloc[0]
        subu_intersection = subu_intersection.iloc[0]
        split_intersection['Training Size'] = subu_intersection['Training Size']
        split_intersection['NO2 MAE Diff'] = split_intersection['NO2 MAE'] - subu_intersection['NO2 MAE']
        split_intersection['O3 MAE Diff'] = split_intersection['O3 MAE'] - subu_intersection['O3 MAE']
        split_intersection['NO2 CvMAE Diff'] = split_intersection['NO2 CvMAE'] - subu_intersection['NO2 CvMAE']
        split_intersection['O3 CvMAE Diff'] = split_intersection['O3 CvMAE'] - subu_intersection['O3 CvMAE']
        merged = merged.append(split_intersection)
        merged = merged.append(subu_intersection)
    return merged

def plot_results(results, column, out_path):
    fig, ax = plt.subplots()
    sns.boxplot(hue='Calibration', y=column, data=results, ax=ax, x='Training Size')
    fig.savefig(out_path, bbox_inches='tight')


@click.command()
@click.argument('out_path', nargs=1)
@click.argument('subu_path', nargs=1)
@click.argument('models', nargs=-1)
def main(out_path, subu_path, models):
    out_path = Path(out_path)
    split_results, split_models = get_results(models)
    subu_results = load_subu(Path(subu_path), split_models)
    merged_results = merge_results(split_results, subu_results)
    merged_results.to_csv(out_path / 'results.csv')
    merged_results.to_latex(out_path / 'results.tex')

    print(merged_results['NO2 CvMAE Diff'].dropna().describe())
    print(merged_results['O3 CvMAE Diff'].dropna().describe())

    plot_results(merged_results, 'NO2 MAE', out_path / 'no2mae.png')
    plot_results(merged_results, 'NO2 CvMAE', out_path / 'no2cvmae.png')
    plot_results(merged_results, 'O3 MAE', out_path / 'o3mae.png')
    plot_results(merged_results, 'O3 CvMAE', out_path / 'o3cvmae.png')

    plot_results(merged_results, 'NO2 MAE Diff', out_path / 'no2mae-diff.png')
    plot_results(merged_results, 'NO2 CvMAE Diff', out_path / 'no2cvmae-diff.png')
    plot_results(merged_results, 'O3 MAE Diff', out_path / 'o3mae-diff.png')
    plot_results(merged_results, 'O3 CvMAE Diff', out_path / 'o3cvmae-diff.png')

if __name__ == '__main__':
    main()

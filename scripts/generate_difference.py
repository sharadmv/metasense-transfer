import ipdb
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
from argparse import ArgumentParser
from collections import OrderedDict

def parse_args():
    argparser = ArgumentParser()
    argparser.add_argument('path')

    return argparser.parse_args()

def load_data(model, path):
    data = pd.read_csv(path)
    return data

MODELS = OrderedDict([
    ('linear', "Linear Regression"),
    ('nn-2', "NN[2]"),
    ('nn-4', "NN[4]"),
    ('subu', "Random Forest")
])

METRICS = ["%s %s"% (gas, metric) for gas in ["NO2", "O3"]
           for metric in ["MAE", "CvMAE", "MSE", "rMSE", "crMSE", "R^2", "MBE"]]

EVALUATIONS = {
    "train": "Train",
    "test": "Test",
    "difference": "Train - Test",
}

if __name__ == "__main__":
    args = parse_args()
    path = Path(args.path)
    out = Path(args.path)
    model_df = pd.DataFrame()
    for result, result_name in EVALUATIONS.items():
        for level in [0, 1, 2, 3]:
            for model, model_name in MODELS.items():
                local_df = load_data(model, path / model / ('level%u' % level) / ('%s.csv' % result))
                local_df = local_df[METRICS]
                local_df['Model'] = model_name
                local_df['Level'] = "Level %u" % level
                local_df['Evaluation'] = result_name
                model_df = pd.concat([model_df, local_df])

    for result, result_name in EVALUATIONS.items():
        for metric in METRICS:
            fig, ax = plt.subplots()
            sns.boxplot(data=model_df[model_df['Evaluation'] == result_name], x="Level", y=metric, hue='Model', whis=2)
            fig.suptitle(result_name)
            fig.savefig(str(out / ('%s_%s.png' % (metric, result))), bbox_inches='tight')
            ax.set_xlabel("")
            plt.close(fig)

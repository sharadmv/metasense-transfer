import itertools
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
    argparser.add_argument('--split', default=None)
    argparser.add_argument('--suffix', default='')
    argparser.add_argument('--ignore-models', nargs='*')

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
current_palette = sns.color_palette()

METRICS = ["%s %s" % (gas, metric) for gas in ["NO2", "O3"]
           for metric in ["crMSE", "MBE"]]

EVALUATIONS = {
    "train": "Train",
    "test": "Test",
    "difference": "Train - Test",
}
markers = ['P', 'o', 'X', 'D', 's']

if __name__ == "__main__":
    args = parse_args()
    path = Path(args.path)
    out = Path(args.path)
    model_df = pd.DataFrame()
    suffix = args.suffix
    if args.suffix:
        suffix = '-' + args.suffix
    for result, result_name in EVALUATIONS.items():
        for level in [0, 1, 2, 3]:
            for model, model_name in MODELS.items():
                local_df = load_data(model, path / model / ('level%u' % level) / ('%s.csv' % result))
                local_df = local_df[METRICS]
                local_df['Model'] = model_name
                local_df['Level'] = "Level %u" % level
                local_df['Evaluation'] = result_name
                model_df = pd.concat([model_df, local_df])
        if result == 'test' and args.split is not None:
            local_df_ = pd.read_csv(args.split)
            # import ipdb; ipdb.set_trace()
            # local_df_ = local_df_[local_df_["Calibration"] == 'Split-NN']
            local_df = local_df_[METRICS]
            local_df['Model'] = "Split-NN"
            local_df['Level'] = "Level 1" #local_df_['Benchmark']
            local_df['Evaluation'] = result_name
            model_df = pd.concat([model_df, local_df])
    if args.split is not None:
        MODELS["Split-NN"] = "Split-NN"

    for gas in ["NO2", "O3"]:
        for result, result_name in EVALUATIONS.items():
            data_f = model_df[model_df['Evaluation'] == result_name]
            x_axis = data_f["%s crMSE" % gas].min(), data_f["%s crMSE" % gas].max()
            y_axis = data_f["%s MBE" % gas].min(), data_f["%s MBE" % gas].max()
            for level in [0, 1, 2, 3]:
                # fig, ax = plt.subplots(1, len(MODELS), sharey=True, sharex=True)
                fig, ax = plt.subplots()
                for i, (model, name) in enumerate(MODELS.items()):
                    if args.ignore_models is not None and model in args.ignore_models:
                        print("Ignoring:", model)
                        continue
                    if model == 'Split-NN' and result != 'test':
                        continue
                    data = model_df[(model_df['Level'] == ("Level %u" % level)) & (model_df['Model'] == name) & (model_df['Evaluation'] == result_name)]
                    # if model == 'Split-NN' and level == 1 and result == 'test':
                        # import ipdb; ipdb.set_trace()
                    ax.scatter(data["%s crMSE" % gas], data["%s MBE" % gas], label=name, alpha=0.6, s=10, marker=markers[i], color=current_palette[i])
                    # ax2[i].scatter(data["%s crMSE" % gas] / data[("epa-%s" % gas).lower()].mean(), data["%s MBE" % gas] / data[("epa-%s" % gas).lower()].mean())
                # ax.set_title(name)
                print("Setting limit:", x_axis, y_axis)
                ax.set_xlim(x_axis)
                ax.set_ylim(y_axis)
                ax.set_xlabel("%s crMSE" % gas)
                ax.set_ylabel("%s MBE" % gas)
                    # ax2[i].set_title(name)
                    # ax2[i].set_xlabel("%s crMSE" % gas)
                    # ax2[i].set_ylabel("%s MBE" % gas)
                ax.legend(loc='best')
                print(str(out / ('%s_level%s_%s_target%s.png' % (gas, level, result, suffix))))
                fig.savefig(str(out / ('%s_level%s_%s_target%s.png' % (gas, level, result, suffix))), bbox_inches='tight', dpi=120)
                # fig2.savefig(str(out / ('%s_level%s_%s_target_norm.png' % (gas, level, result))), bbox_inches='tight')
                plt.close(fig)
    for gas in ["NO2", "O3"]:
        for result, result_name in EVALUATIONS.items():
            data_f = model_df[model_df['Evaluation'] == result_name]
            x_axis = data_f["%s crMSE" % gas].min(), data_f["%s crMSE" % gas].max()
            y_axis = data_f["%s MBE" % gas].min(), data_f["%s MBE" % gas].max()
            for i, (model, name) in enumerate(MODELS.items()):
                if args.ignore_models is not None and model in args.ignore_models:
                    print("Ignoring:", model)
                if model == 'Split-NN' and result != 'test':
                    continue
                fig, ax = plt.subplots()
                for j, level in enumerate([0, 1, 2, 3]):
                    data = model_df[(model_df['Level'] == ("Level %u" % level)) & (model_df['Model'] == name) & (model_df['Evaluation'] == result_name)]
                    ax.scatter(data["%s crMSE" % gas], data["%s MBE" % gas], label="Level %u" % level, alpha=0.6, s=10, marker=markers[j], color=current_palette[i])
                    # ax2[i].scatter(data["%s crMSE" % gas] / data[("epa-%s" % gas).lower()].mean(), data["%s MBE" % gas] / data[("epa-%s" % gas).lower()].mean())
                    fig.suptitle(result_name)
                    ax.set_title(name)
                    ax.set_xlabel("%s crMSE" % gas)
                    ax.set_ylabel("%s MBE" % gas)
                    # ax2[i].set_title(name)
                    # ax2[i].set_xlabel("%s crMSE" % gas)
                    # ax2[i].set_ylabel("%s MBE" % gas)
                print("Setting limit:", x_axis, y_axis)
                ax.set_xlim(x_axis)
                ax.set_ylim(y_axis)
                ax.legend(loc='best')
                fig.savefig(str(out / ('%s_%s_%s_target%s.png' % (gas, model, result, suffix))), bbox_inches='tight', dpi=120)
                # fig2.savefig(str(out / ('%s_level%s_%s_target_norm.png' % (gas, level, result))), bbox_inches='tight')
                plt.close(fig)

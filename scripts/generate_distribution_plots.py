from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict

from metasense.data import load

BOARDS = OrderedDict({
    1: OrderedDict({
        'elcajon': 11,
        'donovan': 19,
        'shafter': 15,
    }),
    2: OrderedDict({
        'elcajon': 17,
        'donovan': 15,
        'shafter': 11,
    }),
    3: OrderedDict({
        'elcajon': 15,
        'donovan': 11,
        'shafter': 17,
    })
})
DATA = {}

ROUND_PLOTS = {
    1: plt.subplots(),
    2: plt.subplots(),
    3: plt.subplots()
}
LOCATION_PLOTS = {
    'elcajon': plt.subplots(),
    'donovan': plt.subplots(),
    'shafter': plt.subplots()
}
ROUND_NO2 = {
    1: plt.subplots(),
    2: plt.subplots(),
    3: plt.subplots()
}
LOCATION_NO2 = {
    'elcajon': plt.subplots(),
    'donovan': plt.subplots(),
    'shafter': plt.subplots()
}
ROUND_O3 = {
    1: plt.subplots(),
    2: plt.subplots(),
    3: plt.subplots()
}
LOCATION_O3 = {
    'elcajon': plt.subplots(),
    'donovan': plt.subplots(),
    'shafter': plt.subplots()
}

for location in LOCATION_PLOTS:
    LOCATION_PLOTS[location][1].set_title("%s - Temperature" % location)
for location in LOCATION_NO2:
    LOCATION_PLOTS[location][1].set_title("%s - Temperature" % location)
for location in LOCATION_O3:
    LOCATION_PLOTS[location][1].set_title("%s - Temperature" % location)
for round in BOARDS:
    ROUND_PLOTS[round][1].set_title("Round %u - Temperature" % round)
    ROUND_NO2[round][1].set_title("Round %u - NO2" % round)
    ROUND_O3[round][1].set_title("Round %u - O3" % round)
    for location in BOARDS[round]:
        data = pd.concat(load(round, location, BOARDS[round][location]))
        temperature = data['temperature'] * 9 / 5 + 32
        no2 = data[data["epa-no2"] < data["epa-no2"].quantile(0.99)]["epa-no2"]
        o3 = data[data["epa-o3"] < data["epa-o3"].quantile(0.99)]["epa-o3"]
        sns.distplot(temperature, ax=ROUND_PLOTS[round][1], label=location, axlabel='Temperature (F)', kde_kws=dict(bw='silverman'), norm_hist=False)
        sns.distplot(temperature, ax=LOCATION_PLOTS[location][1], label="Round %u" % round, axlabel='Temperature (F)', kde_kws=dict(bw='silverman'), norm_hist=False)
        sns.distplot(no2, ax=ROUND_NO2[round][1], label=location, axlabel='NO2 (ppb)', kde_kws=dict(bw='silverman'), norm_hist=False)
        sns.distplot(o3, ax=ROUND_O3[round][1], label=location, axlabel='O3 (ppb)', kde_kws=dict(bw='silverman'), norm_hist=False)
        sns.distplot(no2, ax=LOCATION_NO2[location][1], label="Round %u" % round, axlabel='NO2 (ppb)', kde_kws=dict(bw='silverman'), norm_hist=False)
        sns.distplot(o3, ax=LOCATION_O3[location][1], label="Round %u" % round, axlabel='O3 (ppb)', kde_kws=dict(bw='silverman'), norm_hist=False)
    ROUND_PLOTS[round][1].legend(loc='best')
    ROUND_NO2[round][1].legend(loc='best')
    ROUND_O3[round][1].legend(loc='best')
for location in LOCATION_PLOTS:
    LOCATION_PLOTS[location][1].legend(loc='best')
for location in LOCATION_NO2:
    LOCATION_PLOTS[location][1].legend(loc='best')
for location in LOCATION_O3:
    LOCATION_PLOTS[location][1].legend(loc='best')

out_dir = Path('results') / 'distributions'
out_dir.mkdir(exist_ok=True, parents=True)

for round, plot in ROUND_PLOTS.items():
    plot[0].savefig(str(out_dir / ('round%u_temperature.png' % round)), bbox_inches='tight')
for location, plot in LOCATION_PLOTS.items():
    plot[0].savefig(str(out_dir / ('location_%s_temperature.png' % location)), bbox_inches='tight')
for round, plot in ROUND_NO2.items():
    plot[0].savefig(str(out_dir / ('round%u_no2.png' % round)), bbox_inches='tight')
for round, plot in ROUND_O3.items():
    plot[0].savefig(str(out_dir / ('round%u_o3.png' % round)), bbox_inches='tight')
for location, plot in LOCATION_NO2.items():
    plot[0].savefig(str(out_dir / ('location%s_no2.png' % location)), bbox_inches='tight')
for location, plot in LOCATION_O3.items():
    plot[0].savefig(str(out_dir / ('location%s_o3.png' % location)), bbox_inches='tight')

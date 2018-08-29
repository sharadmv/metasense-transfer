#!/usr/bin/env python
import pandas as pd
from argparse import ArgumentParser

def parse_args():
    argparser = ArgumentParser()

    argparser.add_argument('epa_file')
    argparser.add_argument('csv')
    argparser.add_argument('--max', action='store_true')

    return argparser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    data = pd.read_csv(args.csv, index_col='unixtime')
    data.index = pd.to_datetime(data.index, unit='s')

    minute_data = data
    minute_data.index = pd.to_datetime(minute_data.index).tz_localize("UTC").tz_convert('US/Pacific')
    minute_data = minute_data.sort_index()
    minute_data['nearest'] = minute_data.index.round('min')
    minute_data = minute_data[~minute_data.index.duplicated(keep='first')]

    epa_data = pd.read_csv(args.epa_file, index_col='datetime')
    epa_data.index = pd.to_datetime(epa_data.index).tz_localize('US/Pacific', ambiguous='NaT')
    epa_data = epa_data[epa_data.index.notnull()]
    epa_data = epa_data[~epa_data.index.duplicated(keep='first')]
    epa_data['nearest'] = epa_data.index.round('min', ambiguous=True)

    merged_data = minute_data.merge(epa_data, how='outer', left_index=True, right_index=False, on='nearest', suffixes=('sensor', 'epa')).dropna()
    print(merged_data.to_csv(index_label='datetime'))

#!/usr/bin/env python
import pandas as pd
import sys
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
    # data.drop(['datetime'],axis=1, inplace=True)
    data.index = pd.to_datetime(data.index, unit='s')

    groups = data.groupby(lambda x: "%u/%u/%u %u:%u:00" % (x.year, x.month, x.day, x.hour, x.minute))
    if args.max:
        minute_data = groups.max()
    else:
        minute_data = groups.mean()
    # minute_data = data
    minute_data.index = pd.to_datetime(minute_data.index).tz_localize("UTC").tz_convert('US/Pacific')
    minute_data = minute_data.sort_index()
    # minute_data['nearest'] = minute_data.index.round('min')

    epa_data = pd.read_csv(args.epa_file, index_col='datetime')
    epa_data.index = pd.to_datetime(epa_data.index).tz_localize('US/Pacific', ambiguous=True)
    # epa_data['nearest'] = epa_data.index.round('min')

    # merged_data = minute_data.merge(epa_data, how='outer', left_index=True, right_index=False, on='nearest', suffixes=('sensor', 'epa')).dropna()
    merged_data = minute_data.join(epa_data, how='outer').dropna()
    print(merged_data.to_csv(index_label='datetime'))

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import sys
from os.path import join
from glob import glob
from datetime import datetime


def read_datasets(in_path):
    filenames = glob('krkonose_cloudless_pixels*.csv', root_dir=in_path)
    dfs = {}
    for filename in filenames:
        year = filename[-8:-4]
        df = pd.read_csv(join(in_path, filename), dtype={'doy':'str'})

        df['doy'] = df['doy'].str.pad(3, fillchar='0')
        df['doy'] = pd.to_datetime(df['doy'], format='%j')
        df['undefined'] = df['undefined'].str.replace(',', '').astype('int64')
        df['cloudiness'] = 1 - df['undefined'] / 8_584_350
        dfs[year] = df
    return dfs


def subplot(ax, ds):x
    ax.set_ylabel(ds[0])
    ax.set_yticks([0, 1], labels=['0%', '100%'], fontsize=9)
    vline_xcoors = [datetime(1900, m, 1) for m in [4,5,6,7,8,9,10]]
    ax.vlines(vline_xcoors, colors='grey', ymin=0, ymax=1, linestyles='dashed')
    ax.plot(ds[1]['doy'], ds[1]['cloudiness'], 'o', markersize=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))


def create_plot(dataset_dict):
    fig, axes = plt.subplots(len(dataset_dict.items()), 1, sharex=True)

    for axis, dataset in zip(axes, dataset_dict.items()):
        subplot(axis, dataset)

    fig.suptitle('Sentinel-2 Cloud cover over Krkonoše NP')
    plt.show()


if __name__ == '__main__':
    in_path = r'C:\Users\dd\Documents\NATUR_CUNI\_dp\figures\creation\cloudless_pixels'
    datasets = read_datasets(in_path)
    create_plot(datasets)

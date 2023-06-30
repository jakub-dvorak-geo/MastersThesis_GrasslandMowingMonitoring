
import geopandas as gpd
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import sys
import json
from os.path import join
from pandas import to_datetime
from datetime import datetime


def read_obs_dates(in_path):
    gpkg = gpd.read_file(in_path)
    gpkg['beginposition'] = to_datetime(gpkg['beginposition'])

    obs_dict = {}
    for ro in gpkg['relativeorbitnumber'].unique():
        gpkg_filter = gpkg.loc[gpkg['relativeorbitnumber'] == ro]
        obs_dict[str(ro)] = gpkg_filter['beginposition']
    return obs_dict


def add_subplot(ax, in_situ_dates, obs_dates):
    dates = [datetime.strptime(datestr, '%Y%m%d') for datestr in in_situ_dates]
    for idx, date in enumerate(dates):
        ax.plot([date, date], [-1, 1], linestyle='dashed', color='#63c7ff',
            label='In-Situ observations' if idx==0 else '_nolegend_',
            linewidth=3)

    colors = {'22': '#0c00c1', '73': '#c10054', '95': '#b5c100', '146': '#00c16c'}
    for key in obs_dates.keys():
        dates = obs_dates[key]
        print(dates)
        #dates = [datetime.strptime(datestr, '%Y%m%d') for datestr in obs_dates[key]]
        ax.plot(dates, [[0]]*len(obs_dates[key]), '|', label=key, markersize=20,
            color=colors[key], markeredgewidth=4)

    # Formatting
    ax.set_ylim((-2,2))
    ax.set_yticks([])


    ax.spines[['bottom']].set_position('center')
    ax.spines[['top', 'left', 'right']].set_visible(False)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.tick_params(pad=15)


def create_plot(obs_dates, in_situ_dates):
    # figure creation
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 5))

    # figure styling
    fig.suptitle('Dates of Sentinel-1 observations and field visits')
    ax1.set_title('2021', loc='left', y=0.7)
    ax2.set_title('2022', loc='left', y=0.7)

    # Add data to both subplots
    add_subplot(ax1, in_situ_dates['year_1'], obs_dates['year_1'])
    add_subplot(ax2, in_situ_dates['year_2'], obs_dates['year_2'])

    # Legend Creation
    handles, labels = ax2.get_legend_handles_labels()
    handles.insert(1, Rectangle((0,0), 1, 1, fill=False,
        edgecolor='none', visible=False))
    labels.insert(1, 'Relative Orbits:')
    fig.legend(handles, labels, loc='center', bbox_to_anchor=(0,0,1,0.9),
        ncols=len(labels), frameon=False, handletextpad=0.4)

    plt.show()


if __name__ == '__main__':
    if len(sys.argv) > 2:
        in_path_1 = sys.argv[1]
        in_path_2 = sys.argv[1]
    else:
        root_dir = r'C:\Users\dd\Documents\NATUR_CUNI\_dp\reference_data'

        in_path_1 = join(root_dir, r'Vyjezdy_2021\slc_products_2021.gpkg')
        in_path_2 = join(root_dir, r'Vyjezdy_2022\slc_products_2022.gpkg')

    obs_dates = {
        'year_1': read_obs_dates(in_path_1),
        'year_2': read_obs_dates(in_path_2)}

    print(obs_dates)
    test_obs_dates = {
        'year_1': {'22': ('20210415', '20210427', '20210507', '20210519'),
                   '73': ('20210515', '20210527', '20210607', '20210619'),
                   '95': ('20210615', '20210627', '20210707', '20210719'),
                   '146':('20210715', '20210727', '20210807', '20210819')},
        'year_2': {'22': ('20220415', '20220427', '20220507', '20220519'),
                   '73': ('20220515', '20220527', '20220607', '20220619'),
                   '95': ('20220615', '20220627', '20220707', '20220719'),
                   '146':('20220715', '20220727', '20220807', '20220819')}}
    in_situ_dates = {
        'year_1': ('20210624', '20210708', '20210729', '20210927'),
        'year_2': ('20220811', '20220915')}

    create_plot(obs_dates, in_situ_dates)

import geopandas as gpd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import rasterio
import rasterio.plot

from os.path import join

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def compute_cfar_treshold(x, y):
    # linear regression model
    model = LinearRegression().fit(x, y)

    # model prediciton at the last time step
    coh_fit = model.predict([x[-1]])
    # RMSE of the linear fit
    fit_rmse = mean_squared_error(y, model.predict(x), squared=False)
    # Probability of false alarm
    k = 1
    return coh_fit + k * fit_rmse

def potential_mown_dates(x, y, year=2021):
    df_mown = pd.DataFrame(columns=['date', 'coh', 'confidence'])

    # How many prevous obs should be used for CFAR
    r = 3 if year==2022 else 6

    for idx in range(r, len(x)):
        # create doy converison here
        x_doy = [[int(i.strftime('%j'))] for i in x[idx-r:idx]]
        treshold = compute_cfar_treshold(x_doy, y[idx-r:idx])
        if y[idx] > treshold:
            confidence = 1 - 1 * np.exp(-y[idx])
            df_row = pd.DataFrame([[x[idx],y[idx],confidence]], columns=df_mown.columns)
            df_mown = pd.concat([df_mown, df_row])

    df_mown = df_mown.reset_index(drop=True)
    return df_mown

def identify_mown_dates(x, y, year=2021, min_delta_t=28):
    mown_potential = potential_mown_dates(x, y, year)
    df_mown = pd.DataFrame(columns=['date', 'coh', 'confidence'])
    day_delta = timedelta(days=min_delta_t)

    for _ in range(4):
        if mown_potential.shape[0] > 0:
            # Find detection with max confidence
            max_conf_idx = mown_potential['confidence'].idxmax()
            max_conf_row = mown_potential.loc[max_conf_idx]

            # add the detection to output
            df_mown.loc[len(df_mown)] = max_conf_row

            # Find indices to drop
            condition = (mown_potential['date'] >= (max_conf_row['date'] - day_delta)) & (mown_potential['date'] <= (max_conf_row['date'] + day_delta))
            drop_indices = mown_potential[condition].index

            # drop these given indices from dataFrame
            mown_potential = mown_potential.drop(drop_indices)

    return df_mown


class Time_series:
    def __init__(self, year, relative_orbits=(22,73,95,146)):
        self.year = year
        self.relative_orbits = relative_orbits
        self.sar_gdf = gpd.read_file(join(ROOT_DIR, f'reference_data/{str(year)}/vyjezdy_{year}_zonal_stats.gpkg'))

        self.sar_keys = dict.fromkeys(relative_orbits)
        self.sar_dates = dict.fromkeys(relative_orbits)
        for ro in relative_orbits:
            self.sar_keys[ro] = list(key for key in self.sar_gdf.keys() if f'{ro}_VV_' in key or f'{ro}_VH_' in key)
            self.sar_dates[ro] = list(set([datetime.strptime(key[7:16], '%Y_%m%d') for key in self.sar_keys[ro]]))
            self.sar_dates[ro].sort()
        self.metrics = self._extract_metrics()

        self.obs_gdf = self.sar_gdf.filter(regex='\A\d.*\d\Z', axis='columns')
        self.obs_dates = [datetime.strptime(f'{year}_{date}', '%Y_%d_%m') for date in self.obs_gdf.keys()]

        with rasterio.open(join(ROOT_DIR, 'DTM/DMR_4G_4326.tif')) as src:
            self.dmr = src.read(1)
            self.dmr_transform = src.transform

    def _extract_metrics(self):
        metrics = {}
        for ro in self.relative_orbits:
            pols = set([i[4:6] for i in self.sar_keys[ro]])
            stats = set([i[22:] for i in self.sar_keys[ro]])
            for stat in stats:
                for pol in pols:
                    cols = [coh for coh in self.sar_keys[ro] if stat in coh and pol in coh]
                    cols.sort()
                    metrics[f'{stat}_{pol}_RO-{ro:03}'] = cols
        return metrics

    def list_metrics(self):
        print(self.metrics.keys())

    def _add_twinx_axis(self, ax1, plot_id):
        # Adding Twin Axes
        ax2 = ax1.twinx()
        ax2.set_ylabel('In situ grass height [cm]', color = 'green')
        ax2.set_ylim(0,120)
        plot_2 = ax2.bar(self.obs_dates, self.obs_gdf.iloc[plot_id], width=2, color='green')
        ax2.tick_params(axis ='y', labelcolor = 'green')

    def _add_map_axis(self, ax_map, plot_idx):

        self.sar_gdf.iloc[[plot_idx]].boundary.plot(ax=ax_map)
        xlim, ylim = ax_map.get_xlim(), ax_map.get_ylim()

        rasterio.plot.show(self.dmr, contour=True, ax=ax_map, levels=60, alpha = 0.7,
                           vmin=550, vmax=850, transform=self.dmr_transform, cmap='Greys_r')
        self.sar_gdf.boundary.plot(ax=ax_map, color='black')
        self.sar_gdf.iloc[[plot_idx]].boundary.plot(ax=ax_map, color='green')
        map_buffer = 0.005
        ax_map.set_xlim(xlim[0] - map_buffer, xlim[1] + map_buffer)
        ax_map.set_ylim(ylim[0] - map_buffer, ylim[1] + map_buffer)


    def plot_single_ro(self, plot_idx=0, pol='VH', ro=146, stats=('mean','median','std')):
        #fig, ax1 = plt.subplots()
        fig, (ax1, ax_map) = plt.subplots(1, 2, figsize=(20,5))

        ax1.set_xlabel('Time')
        ax1.set_ylabel('S1 Coherence', color = 'black')
        #ax1.set_ylim(0.25,0.5)
        ax1.tick_params(axis ='y', labelcolor = 'black')

        for stat in stats:
            metric = f'{stat}_{pol}_RO-{ro:03}'
            y_coh = self.sar_gdf.iloc[plot_idx].loc[self.metrics[metric]]
            ax1.plot(self.sar_dates[ro], y_coh, label=stat)
        ax1.legend(loc='upper left')

        # Add twin axis for in-situ measurements
        self._add_twinx_axis(ax1, plot_idx)

        self._add_map_axis(ax_map, plot_idx)

        # Show plot
        plt.show()

    def plot_single_roi(self, plot_idx=0, pol='VH', relative_orbits=(22,73,95,146), stat='mean'):
        fig, (ax1, ax_map) = plt.subplots(1, 2, figsize=(20,5))
        fig.suptitle(f'Coherence {stat} for plot #{plot_idx}')

        ax1.set_xlabel('Time')
        ax1.set_ylabel('S1 Coherence', color = 'black')
        #ax1.set_ylim(0.25,0.5)
        ax1.tick_params(axis ='y', labelcolor = 'black')

        for ro in relative_orbits:
            metric = f'{stat}_{pol}_RO-{ro:03}'
            y_coh = self.sar_gdf.iloc[plot_idx].loc[self.metrics[metric]]
            ax1.plot(self.sar_dates[ro], y_coh, label=ro, marker='+')

            # temp for extraction
            mown = identify_mown_dates(self.sar_dates[ro], y_coh, year=self.year)
            ax1.plot(mown['date'], mown['coh'], marker='o', fillstyle='none', color='black', linestyle='')
        ax1.legend(loc='upper left')

        # Add twin axis for in-situ measurements
        self._add_twinx_axis(ax1, plot_idx)

        self._add_map_axis(ax_map, plot_idx)

        # Show plot
        plt.show()

    def plot_series_all(self, stat='mean', pol='VH', ro=146):
        fig, (axs) = plt.subplots(ncols=6, nrows=10, squeeze=False, sharex='all', sharey='all')
        axs_flat = axs.flatten()
        metric = f'{stat}_{pol}_RO-{ro:03}'

        for idx, row in enumerate(self.sar_gdf.iterrows()):
            x = [datetime.strptime(string[7:16], '%Y_%m%d') for string in row[1][self.metrics[metric]].keys()]
            y = row[1][self.metrics[metric]]
            axs_flat[idx].plot(x, y)


if __name__ == '__main__':
    ROOT_DIR = '/media/sf_JD/DP'
    ROOT_DIR = r'C:\Users\dd\Documents\NATUR_CUNI\_dp'
    year = 2021
    inpath_vector = join(ROOT_DIR, f'reference_data/{year}/vyjezdy_{year}_zonal_stats.gpkg')

    hello_world = Time_series(year)
    for i in range(60):
        hello_world.plot_single_roi(plot_idx=i, relative_orbits=(22,73,95,146), stat='median')

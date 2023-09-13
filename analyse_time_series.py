import geopandas as gpd
import matplotlib.pyplot as plt
from datetime import date, datetime, timedelta
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
    k = 1#3e-7
    return coh_fit, k * fit_rmse

def potential_mown_dates(x, y, year=2021):
    df_mown = pd.DataFrame(columns=['date', 'coh', 'confidence'])

    # How many prevous obs should be used for CFAR
    r = 3 if year==2022 else 6

    for idx in range(r, len(x)):
        # create doy converison here
        x_doy = [[int(i.strftime('%j'))] for i in x[idx-r:idx]]
        fit_prev_idx, false_alarm = compute_cfar_treshold(x_doy, y[idx-r:idx])
        if y[idx] > (fit_prev_idx + false_alarm):
            confidence = 1 - 1 * np.exp(-(y[idx]-fit_prev_idx))
            df_row = pd.DataFrame([[x[idx],y[idx],confidence[0]]], columns=df_mown.columns)
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
        self.orig_gdf = gpd.read_file(join(ROOT_DIR, f'reference_data/{str(year)}/vyjezdy_{year}_4326_singlepart.gpkg'))

        self.sar_keys = dict.fromkeys(relative_orbits)
        self.sar_dates = dict.fromkeys(relative_orbits)
        for ro in relative_orbits:
            self.sar_keys[ro] = list(key for key in self.sar_gdf.keys() if f'{ro}_VV_' in key or f'{ro}_VH_' in key)
            self.sar_dates[ro] = list(set([datetime.strptime(key[7:16], '%Y_%m%d') for key in self.sar_keys[ro]]))
            self.sar_dates[ro].sort()
        self.metrics = self._extract_metrics()
        self.detections = {}

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
        # Adding potential mowing date ranges
        plot_reference_mowdates(ax2, plot_id)
        # Adding bars for in-situ measurements
        plot_2 = ax2.bar(self.obs_dates, self.obs_gdf.iloc[plot_id], width=2, color='green')
        ax2.tick_params(axis ='y', labelcolor = 'green')

        # Adding detected ranges
        plot_sentinel_mowdates(ax2, self.detections[plot_id])

    def _add_map_axis(self, ax_map, plot_idx):

        self.orig_gdf.iloc[[plot_idx]].boundary.plot(ax=ax_map)
        xlim, ylim = ax_map.get_xlim(), ax_map.get_ylim()

        rasterio.plot.show(self.dmr, contour=True, ax=ax_map, levels=60, alpha = 0.7,
                           vmin=550, vmax=850, transform=self.dmr_transform, cmap='Greys_r')
        self.orig_gdf.boundary.plot(ax=ax_map, color='#00008b')
        self.orig_gdf.iloc[[plot_idx]].boundary.plot(ax=ax_map, color='#4cbb17')
        map_buffer = 0.005
        ax_map.set_xlim(xlim[0] - map_buffer, xlim[1] + map_buffer)
        ax_map.set_ylim(ylim[0] - map_buffer, ylim[1] + map_buffer)
        ax_map.tick_params(axis='x', labelrotation=45)
        ax_map.ticklabel_format(style='plain')


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
        fig, (ax1, ax_map) = plt.subplots(1, 2, figsize=(14,5.5), layout='tight')
        fig.suptitle(f'{pol} coherence {stat} for plot #{plot_idx}')

        ax1.set_xlabel('Time')
        ax1.set_ylabel('S1 Coherence', color = 'black')
        ax1.tick_params(axis ='y', labelcolor = 'black')
        self.detections[plot_idx] = {}

        for ro in relative_orbits:
            metric = f'{stat}_{pol}_RO-{ro:03}'
            y_coh = self.sar_gdf.iloc[plot_idx].loc[self.metrics[metric]]
            ax1.plot(self.sar_dates[ro], y_coh, label=ro, marker='+')

            # temp for extraction
            mown = identify_mown_dates(self.sar_dates[ro], y_coh, year=self.year)
            ax1.plot(mown['date'], mown['coh'], marker='o', fillstyle='none', color='black', linestyle='')
            self.detections[plot_idx][ro] = mown
        ax1.legend(loc='upper left')
        ax1.set_ylim([0,.6])

        # Add twin axis for in-situ measurements
        self._add_twinx_axis(ax1, plot_idx)
        self._add_map_axis(ax_map, plot_idx)

    def plot_single_roi_both_pol(self, plot_idx=0, relative_orbits=(22,73,95,146), stat='mean'):
        fig, (ax1, ax_map) = plt.subplots(1, 2, figsize=(14,5.5), layout='tight')
        fig.suptitle(f'Coherence {stat} for plot #{plot_idx} (both VH and VV)')

        ax1.set_xlabel('Time')
        ax1.set_ylabel('S1 Coherence', color = 'black')
        ax1.tick_params(axis ='y', labelcolor = 'black')
        self.detections[plot_idx] = {}

        for ro in relative_orbits:
            mown_both_pol = []
            for pol in ('VH', 'VV'):
                metric = f'{stat}_{pol}_RO-{ro:03}'
                y_coh = self.sar_gdf.iloc[plot_idx].loc[self.metrics[metric]]
                ax1.plot(self.sar_dates[ro], y_coh, label=f'{pol}_{ro}', marker='+')

                # temp for extraction
                mown = identify_mown_dates(self.sar_dates[ro], y_coh, year=self.year)
                ax1.plot(mown['date'], mown['coh'], marker='o', fillstyle='none', color='black', linestyle='')
                #self.detections[plot_idx][ro] = mown
                mown_both_pol.append(mown)
            self.detections[plot_idx][ro] = pd.concat(mown_both_pol, ignore_index=True)

        ax1.legend(loc='upper left')
        ax1.set_ylim([0,.6])

        # Add twin axis for in-situ measurements
        self._add_twinx_axis(ax1, plot_idx)
        self._add_map_axis(ax_map, plot_idx)


    def plot_series_all(self, stat='mean', pol='VH', ro=146):
        fig, (axs) = plt.subplots(ncols=6, nrows=10, squeeze=False, sharex='all', sharey='all')
        axs_flat = axs.flatten()
        metric = f'{stat}_{pol}_RO-{ro:03}'

        for idx, row in enumerate(self.sar_gdf.iterrows()):
            x = [datetime.strptime(string[7:16], '%Y_%m%d') for string in row[1][self.metrics[metric]].keys()]
            y = row[1][self.metrics[metric]]
            axs_flat[idx].plot(x, y)


def read_heights(in_path):
    def _str_to_date(in_str):
        day, month = in_str.split('_')
        return date(int(year), int(month), int(day))

    year = in_path.split('.')[0][-16:-12]
    in_gdf = gpd.read_file(in_path)

    gdf_in_situ = in_gdf.filter(regex='\A\d.*\d\Z', axis='columns')
    gdf_as_dates = gdf_in_situ.rename(columns=_str_to_date)
    gdf_sorted = gdf_as_dates[sorted(gdf_as_dates)]

    return gdf_sorted

def overall_mown_date(gdf, threshold=20):
    cols = gdf.columns

    # first date
    col_0 = gdf[cols[0]]
    conditions = [
        (col_0 <= 10),
        (col_0 <= 20) & (col_0 > 10)
        ]
    values = [
        cols[0] - timedelta(days=14),
        cols[0] - timedelta(days=21)
        ]
    gdf[f'possible_mow_date_before_{cols[0]}'] = np.select(conditions, values)

    # all other dates
    for idx in range(len(cols) - 1):
        condition_mown = (gdf[cols[idx]] >= gdf[cols[idx+1]] + threshold)
        col_1 = gdf[cols[idx+1]]
        days_since_last = (cols[idx+1]-cols[idx]).days
        conditions = [
            condition_mown & (col_1 <= 20),
            condition_mown & (col_1 <= 40) & (col_1 > 20)
            ]
        values = [
            cols[idx+1] - timedelta(days=min(128, days_since_last)),
            cols[idx+1] - timedelta(days=min(128, days_since_last))
            ]
        gdf[f'possible_mow_date_before_{cols[idx+1]}'] = np.select(conditions, values)
    return gdf

def plot_one(ax, gdf, id=4):
    gid = gdf.loc[id,:]
    cols = gdf.columns

    max_val = gid[:int((len(cols)+1)/2)].max()
    max_val = 10

    for idx in range(int((len(cols))/2)):
        start = gid.loc[cols[idx+int((len(cols)+1)/2)]]
        end = cols[idx]
        #x_vals = [gid[cols[idx]]]
        #print(x_vals)
        if start != 0:
            x_vals = np.array([start, end, end, start], dtype=np.datetime64)
            y_vals = [0, 0, max_val, max_val]
            #ax.fill_between()
            ax.fill(x_vals, y_vals, alpha=0.25, color='#a8e4a0')

def plot_reference_mowdates(ax, id):
    gdf_heights = read_heights(inpath_vector)
    gdf_mowdates = overall_mown_date(gdf_heights, threshold=10)
    plot_one(ax, gdf_mowdates, id)


def assign_confidences(df):
    for idx_0 in range(len(df)):
        df.loc[idx_0, 'detection_count'] = 0
        for idx_1 in range(len(df)):
            if df.loc[idx_0, 'date'] <= df.loc[idx_1, 'date'] < (df.loc[idx_0, 'date'] + timedelta(days=12)):
                df.loc[idx_0, 'detection_count'] += 1
    df = df.sort_values(by='detection_count', ascending=False)


    out_dates = pd.DataFrame(columns=['date', 'detection_count'])
    while len(df) > 0:
        df = df.reset_index(drop=True)
        out_dates = pd.concat([out_dates, df.loc[0, ['date', 'detection_count']].to_frame().T], ignore_index=True)

        df = df.drop(
            df[
                (df.loc[:, 'date'] >= df.loc[0, 'date']) &
                (df.loc[:, 'date'] < (df.loc[0, 'date'] + timedelta(days=12)))
            ].index)

    return out_dates


def plot_mowdates(ax, gdf):
    # filter out detections with only one RO
    gdf = gdf[gdf['detection_count'] > 1]
    for _, row in gdf.iterrows():
        start = row['date'] - timedelta(days=12)
        end = row['date']
        x_vals = np.array([start, end, end, start], dtype=np.datetime64)
        y_vals = [0, 0, 5, 5]
        print(row['detection_count'])
        if row['detection_count'] == 8:
            color = 'green'
        elif row['detection_count'] == 7:
            color = 'yellow'
        elif row['detection_count'] == 6:
            color = 'orange'
        elif row['detection_count'] == 5:
            color = 'red'
        elif row['detection_count'] == 4:
            color = 'purple'

        if row['detection_count'] > 3:
            ax.fill(x_vals, y_vals, alpha=0.5, color=color)


def plot_sentinel_mowdates(ax, df_dict):
    dates_combined = pd.DataFrame(columns=['date', 'coh', 'confidence', 'detection_count'])
    for value in df_dict.values():
        dates_combined = pd.concat([dates_combined, value])
    dates_combined = dates_combined.sort_values(by='date')
    dates_combined = dates_combined.reset_index(drop=True)

    confidence_detections = assign_confidences(dates_combined)
    plot_mowdates(ax, confidence_detections)


if __name__ == '__main__':
    ROOT_DIR = '/media/sf_JD/DP'
    ROOT_DIR = r'C:\Users\dd\Documents\NATUR_CUNI\_dp'
    year = 2021
    inpath_vector = join(ROOT_DIR, f'reference_data/{year}/vyjezdy_{year}_zonal_stats.gpkg')

    hello_world = Time_series(year)
    print(hello_world.sar_gdf.shape)

    #hello_world.plot_single_roi(plot_idx=54, relative_orbits=(22,73,95,146), stat='median', pol='VH')
    #hello_world.plot_single_roi(plot_idx=54, relative_orbits=(22,73,95,146), stat='median', pol='VV')
    hello_world.plot_single_roi_both_pol(plot_idx=50, relative_orbits=(22,73,95,146), stat='median')
    plt.show()
    """
    for i in range(61):
        #hello_world.plot_single_roi(plot_idx=i, relative_orbits=(22,73,95,146), stat='median', pol='VH')
        #hello_world.plot_single_roi(plot_idx=i, relative_orbits=(22,73,95,146), stat='median', pol='VV')
        hello_world.plot_single_roi_both_pol(plot_idx=i, relative_orbits=(22,73,146), stat='median')
        # Show plot
        plt.show()
    """

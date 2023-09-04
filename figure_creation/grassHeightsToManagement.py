import geopandas as gpd
import matplotlib.pyplot as plt
from datetime import date, timedelta
import numpy as np
import pandas as pd


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

def extract_coh(in_path, ro, pol, statistic, id):
    def _str_to_date(in_str):
        datestrs = in_str.split('_')[3].split('-')
        m_1, d_1 = int(datestrs[0][0:2]), int(datestrs[0][2:4])
        m_2, d_2 = int(datestrs[1][0:2]), int(datestrs[1][2:4])
        return date(year, m_1, d_1), date(year, m_2, d_2)

    filter = f'{ro}_{pol}.*{statistic}'

    year = int(in_path.split('.')[0][-16:-12])
    in_gdf = gpd.read_file(in_path)
    gdf_filter = in_gdf.filter(regex=filter, axis='columns')
    print(gdf_filter)
    gdf_id = gdf_filter.loc[id]
    print(gdf_id)

    individual_dfs = []
    for col in gdf_filter.columns:
        dates = _str_to_date(col)
        dates = _str_to_date(col)
        df_col = pd.DataFrame({
            'coh': [gdf_id[col]],
            'date1': dates[0],
            'date2': dates[1]
            })
        #df_col['date1'], df_col['date2'] = _str_to_date(col)
        print(df_col)
        individual_dfs.append(df_col)

    df_out = pd.concat(individual_dfs)


    #gdf_filter['date1'], gdf_filter['date2'] =
    gdf_as_dates = gdf_filter.rename(columns=_str_to_date)
    series = gdf_as_dates.loc[id]

    return df_out


def plot_time_series(gdf):
    gdf.transpose().plot()
    plt.show()

def overall_mown_date(gdf, threshold=20):
    #conditions, values = [], []

    cols = gdf.columns
    for idx in range(len(cols) - 1):
        condition_mown = (gdf[cols[idx]] > gdf[cols[idx+1]] + threshold)
        col_1 = gdf[cols[idx+1]]
        days_since_last = (cols[idx+1]-cols[idx]).days
        conditions = [
            condition_mown & (col_1 <= 10),
            condition_mown & (col_1 <= 40) & (col_1 > 20)
            ]
        values = [
            cols[idx+1] - timedelta(days=min(14, days_since_last)),
            cols[idx+1] - timedelta(days=min(28, days_since_last))
            ]
        gdf[f'possible_mow_date_before_{cols[idx+1]}'] = np.select(conditions, values)
    #gdf['first_possible_mow_date'] = np.select(conditions, values)
    return gdf

def plot_one(gdf, coh, id=4):
    gid = gdf.loc[id,:]
    cols = gdf.columns

    plt.hlines(y='coh', xmin='date1', xmax='date2', data=coh)
    ax = plt.twinx()

    ax.plot(gid[:int((len(gdf.columns)+1)/2)])
    max_val = gid[:int((len(gdf.columns)+1)/2)].max()

    for idx in range(int((len(gdf.columns)-1)/2)):
        start = gid.loc[cols[idx+int((len(gdf.columns)+1)/2)]]
        end = cols[idx+1]
        #x_vals = [gid[cols[idx]]]
        #print(x_vals)
        if start != 0:
            x_vals = np.array([start, end, end, start], dtype=np.datetime64)
            y_vals = [0, 0, max_val, max_val]
            #ax.fill_between()
            ax.fill(x_vals, y_vals, alpha=0.25)
    #ax.ylim(0, 100)
    plt.show()


def was_mown(gdf, threshold=20):
    figure, axes = plt.subplots(len(gdf.columns)-1, sharex=True)
    cols = gdf.columns
    if len(cols) == 2:
        axes = [axes]

    out_dict = {}
    for idx, ax in enumerate(axes):
        gdf_filtered = gdf[gdf[cols[idx]] >= gdf[cols[idx+1]]]
        gdf_filtered_threshold = gdf[gdf[cols[idx]] > gdf[cols[idx+1]] + threshold]
        datedelta = cols[idx+1] - cols[idx]
        out_dict[f'{cols[idx]} to {cols[idx+1]}'] = [datedelta, gdf_filtered_threshold]

        ax.plot(gdf_filtered_threshold.transpose())
        ax.set_title(f'There are {gdf_filtered.shape[0]} plots with a decrease between {cols[idx]} and {cols[idx+1]}\n{gdf_filtered_threshold.shape[0]} of them decreased by over {threshold}cm')
    #plt.show()
    return out_dict


def when_mown(filtered_dict):
    out_dict = {}
    for idx, datepair in enumerate(filtered_dict.keys()):
        cols = filtered_dict[datepair][1].columns
        gdf_filter = filtered_dict[datepair][1][filtered_dict[datepair][1][cols[idx+1]] <= 40]

        days_since_last = filtered_dict[datepair][0].days

        conditions = [
            (gdf_filter[cols[idx+1]] <= 20),
            (gdf_filter[cols[idx+1]] > 20)
            ]
        values = [min(14, days_since_last), min(28, days_since_last)]

        gdf_filter['days_since_mown'] = np.select(conditions, values)
        out_dict[datepair] = gdf_filter
    return out_dict


def plot_overall(in_dict):
    dates = in_dict.keys()
    fig, axes = plt.subplots(len(dates))

    if len(dates) == 1:
        axes = [axes]

    for ax, date_period in zip(axes, dates):
        ax.plot(in_dict[date_period]['< 14 days'].transpose())
    plt.show()


def plot_multiple_years(paths_list):
    for in_path in paths_list:
        gdf_heights = read_heights(in_path)
        gdf_mowdates = overall_mown_date(gdf_heights)
        """
        for id in range(gdf_mowdates.shape[0]):
            coh_146_VH_mean = extract_coh(in_path, '146', 'VH', 'mean', id)
            plot_one(gdf_mowdates, coh_146_VH_mean, id)
        """
        id = 4
        coh_146_VH_mean = extract_coh(in_path, '146', 'VH', 'mean', id)
        plot_one(gdf_mowdates, coh_146_VH_mean, id)

        #plot_time_series(gdf_heights)
        #plots_with_decrease = was_mown(gdf_heights)
        #when_mown_dict = when_mown(plots_with_decrease)
        #print(when_mown_dict)
        #plot_overall(when_mown_dict)


if __name__ == '__main__':
    in_vectors = [r'C:\Users\dd\Documents\NATUR_CUNI\_dp\reference_data\2021\vyjezdy_2021_zonal_stats.gpkg',
        r'C:\Users\dd\Documents\NATUR_CUNI\_dp\reference_data\2022\vyjezdy_2022_zonal_stats.gpkg']

    plot_multiple_years([in_vectors[0]])

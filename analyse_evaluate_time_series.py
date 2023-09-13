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


class Time_series:
    def __init__(self, year, relative_orbits=(22,73,95,146), pols=('VH', 'VV')):
        self.pols = pols
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

        self.obs_gdf = self.sar_gdf.filter(regex='\A\d.*\d\Z', axis='columns')
        self.obs_dates = [datetime.strptime(f'{year}_{date}', '%Y_%d_%m') for date in self.obs_gdf.keys()]

        self.grass_heights = self._read_grass_heights()
        self._identify_in_situ_detections()

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

    def _read_grass_heights(self):
        def _str_to_date(in_str):
            day, month = in_str.split('_')
            return date(int(year), int(month), int(day))

        gdf_in_situ = self.sar_gdf.filter(regex='\A\d.*\d\Z', axis='columns')
        gdf_as_dates = gdf_in_situ.rename(columns=_str_to_date)
        gdf_sorted = gdf_as_dates[sorted(gdf_as_dates)]
        return gdf_sorted

    def _identify_in_situ_detections(self, threshold=10):
        cols = self.grass_heights.columns
        self.in_situ_detection_dates = pd.DataFrame(columns=cols)

        # first date
        col_0 = self.grass_heights[cols[0]]
        conditions = [(col_0 <= 10), (col_0 <= 20) & (col_0 > 10)]
        values = [cols[0] - timedelta(days=14), cols[0] - timedelta(days=21)]
        self.in_situ_detection_dates[cols[0]] = np.select(conditions, values, default=np.nan)

        # all later dates
        for idx in range(len(cols) - 1):
            condition_mown = (self.grass_heights[cols[idx]] >= self.grass_heights[cols[idx+1]] + threshold)
            days_since_last = (cols[idx+1]-cols[idx]).days
            start_date = cols[idx+1] - timedelta(days=days_since_last)
            self.in_situ_detection_dates[cols[idx+1]] = np.where(condition_mown, start_date, np.nan)

    def _compute_cfar_treshold(self, x, y):
        # linear regression model
        model = LinearRegression().fit(x, y)

        # model prediciton at the last time step
        coh_fit = model.predict([x[-1]])
        # RMSE of the linear fit
        fit_rmse = mean_squared_error(y, model.predict(x), squared=False, multioutput='raw_values')
        # Probability of false alarm
        return coh_fit, self.cfar_k * fit_rmse

    def _potential_mown_dates(self, x, y):
        def _str_to_date(in_str):
            year, month, day = in_str[7:11], in_str[12:14], in_str[14:16]
            return date(int(year), int(month), int(day))

        confidences = []
        # How many prevous obs should be used for CFAR
        r = 3 if self.year==2022 else 6

        for idx in range(r, len(x)):
            # create doy converison here
            x_doy = [[int(i.strftime('%j'))] for i in x[idx-r:idx]]
            fit_prev_idx, false_alarm = self._compute_cfar_treshold(x_doy, y.iloc[:, idx-r:idx].T)
            confidence = 1 - 1 * np.exp(-(y.iloc[:,idx]-fit_prev_idx[0,:]))
            confidences.append(confidence[y.iloc[:, idx] > (fit_prev_idx[0,:] + false_alarm)])

        return pd.concat(confidences, axis=1).sort_index().rename(_str_to_date, axis=1)


    def identify_mown_dates(self, stat='median', cfar_k=3e-7, min_delta_t=28):
        day_delta = timedelta(days=min_delta_t)
        self.cfar_k = cfar_k

        self.s1_mown_dates = {}
        for pol in self.pols:
            self.s1_mown_dates[pol] = {}
            for ro in self.relative_orbits:
                self.s1_mown_dates[pol][ro] = pd.DataFrame(columns=[0, 1, 2, 3])
                x = self.sar_dates[ro]
                y = self.sar_gdf.loc[:,self.metrics[f'{stat}_{pol}_RO-{ro:03}']]
                mown_potential = self._potential_mown_dates(x, y)
                for idx in self.s1_mown_dates[pol][ro].columns:
                    # Find detection with max confidence
                    detection_idx = mown_potential.idxmax(axis=1)
                    # Add it to out df
                    self.s1_mown_dates[pol][ro][idx] = detection_idx
                    # Find indices to drop

                    for potential_dates, detect_date in zip(mown_potential.iterrows(), detection_idx):
                        idx_row = potential_dates[0]
                        potential_dates = potential_dates[1].index
                        if pd.notna(detect_date):
                            condition = (potential_dates >= detect_date - day_delta) & (potential_dates <= detect_date + day_delta)
                            mown_potential.iloc[idx_row,:][condition] = None

        return self.s1_mown_dates


    def _validate_dates(self, s1_detection_period=6):
        df_field = self.in_situ_detection_dates
        total_field = df_field.count(axis=1)
        self.validation = {}
        for pol in self.pols:
            self.validation[pol] = {}
            for ro in self.relative_orbits:
                self.validation[pol][ro] = pd.DataFrame(columns=('TP','FP','FN'))
                df_s1 = self.s1_mown_dates[pol][ro]
                total_s1 = df_s1.count(axis=1)
                #print(df_s1)
                #print(total_s1)
                #print(df_field)
                #print(total_field)
                for (plot_idx, row_s1), (_, row_field) in zip(df_s1.iterrows(), df_field.iterrows()):
                    row_s1_end = row_s1.dropna()
                    row_s1_start = row_s1_end - timedelta(s1_detection_period)
                    #print(row_s1_start)
                    #print(row_s1_end)
                    row_field_start = row_field.dropna()
                    row_field_end = row_field_start.index
                    #print(row_field_start)
                    #print(row_field_end)
                    self.validation[pol][ro].loc[plot_idx] = [0, 0, 0]
                    arr = np.zeros((len(row_s1_end), len(row_field_end)))
                    for idx_field, (field_start, field_end) in enumerate(zip(row_field_start, row_field_end)):
                        for idx_s1, (s1_start, s1_end) in enumerate(zip(row_s1_start, row_s1_end)):
                            check_intersect = (s1_start < field_end) & (s1_end > field_start)
                            if check_intersect:
                                #self.validation[pol][ro]['TP'].loc[plot_idx] += 1
                                arr[idx_s1, idx_field] += 1
                            #print(field_start, field_end)
                            #print(s1_start, s1_end)
                            #print(check_intersect)
                    if arr.sum(axis=0).max() == 1 and arr.sum(axis=1).max() == 1:
                        self.validation[pol][ro]['TP'].loc[plot_idx] = arr.sum()
                    elif arr.sum(axis=0).max() == 1 and arr.sum(axis=1).max() > 1:
                        arr_row_sum_0 = arr.sum(axis=1) - 1
                        arr_row_sum_0[arr_row_sum_0 < 0] = 0
                        self.validation[pol][ro]['TP'].loc[plot_idx] = arr.sum() - arr_row_sum_0.sum()
                    elif arr.sum(axis=0).max() > 1 and arr.sum(axis=1).max() == 1:
                        arr_col_sum_0 = arr.sum(axis=0) - 1
                        arr_col_sum_0[arr_col_sum_0 < 0] = 0
                        self.validation[pol][ro]['TP'].loc[plot_idx] = arr.sum() - arr_col_sum_0.sum()
                    else:
                        print('ISSUE, STRANGE VALUES IN DETECTION ALG')
                        print(f'Plot idx: {plot_idx}')
                        print(arr)
                        print('arr_sum')
                        print(arr.sum())
                        print('arr_col_sum')
                        print(arr.sum(axis=0))
                        print('arr_row_sum')
                        print(arr.sum(axis=1))
                        print('------')

                    self.validation[pol][ro]['FP'].loc[plot_idx] = total_s1.loc[plot_idx] - self.validation[pol][ro]['TP'].loc[plot_idx]
                    self.validation[pol][ro]['FN'].loc[plot_idx] = total_field.loc[plot_idx] - self.validation[pol][ro]['TP'].loc[plot_idx]
        return self.validation

    def evaluate_date_intersects(self, s1_detection_period=6):
        self._validate_dates(s1_detection_period=s1_detection_period)

        for pol in self.pols:
            for ro in self.relative_orbits:
                tp = self.validation[pol][ro]['TP'].sum()
                fp = self.validation[pol][ro]['FP'].sum()
                fn = self.validation[pol][ro]['FN'].sum()

                ua = tp / (tp + fp)
                pa = tp / (tp + fn)
                f1 = 2 * (ua * pa) / (ua + pa)
                print(pol, ro)
                print('TP, FP, FN')
                print(tp, fp, fn)
                print('UA, PA, F1')
                print(ua, pa, f1)
                print('-------------------------')

    def export_validation_df(self, out_dir):
        for ro in self.relative_orbits:
            for pol in self.pols:
                out_df = self.validation[pol][ro]
                out_path = join(out_dir, f'validated_py_{pol}_{ro:03}.csv')
                out_df.to_csv(out_path, sep=';')


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    ROOT_DIR = '/media/sf_JD/DP'
    ROOT_DIR = r'C:\Users\dd\Documents\NATUR_CUNI\_dp'
    year = 2021

    ts = Time_series(year)

    ts.identify_mown_dates(cfar_k=1)
    ts.evaluate_date_intersects(s1_detection_period=12)
    ts.export_validation_df(join(ROOT_DIR, 'results'))

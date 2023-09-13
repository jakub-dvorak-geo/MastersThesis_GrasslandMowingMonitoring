import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

def read_reference(in_path, in_dmr):
    gdf = gpd.read_file(in_path)
    gdf_dmr = gpd.read_file(in_dmr)
    return gdf.merge(gdf_dmr, left_index=True, right_index=True)

def read_validated(in_path):
    df = pd.read_csv(in_path, sep=';')
    return df

def plot_metric(df, level='MEDIUM', x_lab='area', y_lab='F1'):

    fig, (ax1, ax2) = plt.subplots(1,2)
    x = df[x_lab]
    vh_tp = df[f'{level}_VH_TP']
    vh_fp = df[f'{level}_VH_FP']
    vh_fn = df[f'{level}_VH_FN']
    vv_tp = df[f'{level}_VV_TP']
    vv_fp = df[f'{level}_VV_FP']
    vv_fn = df[f'{level}_VV_FN']

    vh_ua = vh_tp / (vh_tp + vh_fp)
    vv_ua = vv_tp / (vv_tp + vv_fp)
    vh_pa = vh_tp / (vh_tp + vh_fn)
    vv_pa = vv_tp / (vv_tp + vv_fn)

    if y_lab == 'UA':
        ax1.scatter(x, vh_ua)
        ax2.scatter(x, vv_ua)
    if y_lab == 'PA':
        ax1.scatter(x, vh_pa)
        ax2.scatter(x, vv_pa)
    if y_lab == 'F1':
        vh_f1 = 2 * (vh_ua * vh_pa) / (vh_ua + vh_pa)
        vv_f1 = 2 * (vv_ua * vv_pa) / (vv_ua + vv_pa)

        ax1.scatter(x, vh_f1)
        ax2.scatter(x, vv_f1)

    plt.show()

def group_by_area(df, level='HIGH'):
    out_list = []
    for i in range(8):
        min_val = i * 10000
        max_val = (i+1) * 10000
        df_filter = df[(df['area'] > min_val) & (df['area'] <= max_val)]
        vh_tp = df_filter[f'{level}_VH_TP'].sum()
        vh_fp = df_filter[f'{level}_VH_FP'].sum()
        vh_fn = df_filter[f'{level}_VH_FN'].sum()
        vv_tp = df_filter[f'{level}_VV_TP'].sum()
        vv_fp = df_filter[f'{level}_VV_FP'].sum()
        vv_fn = df_filter[f'{level}_VV_FN'].sum()

        vh_ua = vh_tp / (vh_tp + vh_fp)
        vv_ua = vv_tp / (vv_tp + vv_fp)
        vh_pa = vh_tp / (vh_tp + vh_fn)
        vv_pa = vv_tp / (vv_tp + vv_fn)

        vh_f1 = 2 * (vh_ua * vh_pa) / (vh_ua + vh_pa)
        vv_f1 = 2 * (vv_ua * vv_pa) / (vv_ua + vv_pa)

        print(f'{min_val} < Area < {max_val}')
        print('VH')
        print('UA, PA, F1')
        print(vh_ua, vh_pa, vh_f1)
        print('VV')
        print('UA, PA, F1')
        print(vv_ua, vv_pa, vv_f1)
        print('----------------------')

def group_by_area(df, level='HIGH'):
    out_list = []
    for i in range(8):
        min_val = i * 10000
        max_val = (i+1) * 10000
        df_filter = df[(df['area'] > min_val) & (df['area'] <= max_val)]
        vh_tp = df_filter[f'{level}_VH_TP'].sum()
        vh_fp = df_filter[f'{level}_VH_FP'].sum()
        vh_fn = df_filter[f'{level}_VH_FN'].sum()
        vv_tp = df_filter[f'{level}_VV_TP'].sum()
        vv_fp = df_filter[f'{level}_VV_FP'].sum()
        vv_fn = df_filter[f'{level}_VV_FN'].sum()

        vh_ua = vh_tp / (vh_tp + vh_fp)
        vv_ua = vv_tp / (vv_tp + vv_fp)
        vh_pa = vh_tp / (vh_tp + vh_fn)
        vv_pa = vv_tp / (vv_tp + vv_fn)

        vh_f1 = 2 * (vh_ua * vh_pa) / (vh_ua + vh_pa)
        vv_f1 = 2 * (vv_ua * vv_pa) / (vv_ua + vv_pa)

        print(f'{min_val} < Area < {max_val}')
        print('VH')
        print('UA, PA, F1')
        print(vh_ua, vh_pa, vh_f1)
        print('VV')
        print('UA, PA, F1')
        print(vv_ua, vv_pa, vv_f1)
        print('----------------------')

def group_by_area_new(df, level='5'):
    for i in range(8):
        min_val = i * 10000
        max_val = (i+1) * 10000
        df_filter = df[(df['area'] > min_val) & (df['area'] <= max_val)]
        tp = df_filter[f'TP_{level}'].sum()
        fp = df_filter[f'FP_{level}'].sum()
        fn = df_filter[f'FN_{level}'].sum()

        ua = tp / (tp + fp)
        pa = tp / (tp + fn)

        f1 = 2 * (ua * pa) / (ua + pa)

        print(f'{min_val} < Area < {max_val}')
        print(df_filter['area'])
        print('UA, PA, F1')
        print(ua, pa, f1)
        print('----------------------')

def group_by_aspect(df, level='HIGH'):
    out_list = []
    dfs = [
        df[(df['ASPECT_mean'] <= 45) | (df['ASPECT_mean'] > 315)],
        df[(df['ASPECT_mean'] > 45) & (df['ASPECT_mean'] <= 135)],
        df[(df['ASPECT_mean'] > 135) & (df['ASPECT_mean'] <= 225)],
        df[(df['ASPECT_mean'] > 225) & (df['ASPECT_mean'] <= 315)]
    ]
    print(dfs)

    for df_filter in dfs:
        vh_tp = df_filter[f'{level}_VH_TP'].sum()
        vh_fp = df_filter[f'{level}_VH_FP'].sum()
        vh_fn = df_filter[f'{level}_VH_FN'].sum()
        vv_tp = df_filter[f'{level}_VV_TP'].sum()
        vv_fp = df_filter[f'{level}_VV_FP'].sum()
        vv_fn = df_filter[f'{level}_VV_FN'].sum()

        vh_ua = vh_tp / (vh_tp + vh_fp)
        vv_ua = vv_tp / (vv_tp + vv_fp)
        vh_pa = vh_tp / (vh_tp + vh_fn)
        vv_pa = vv_tp / (vv_tp + vv_fn)

        vh_f1 = 2 * (vh_ua * vh_pa) / (vh_ua + vh_pa)
        vv_f1 = 2 * (vv_ua * vv_pa) / (vv_ua + vv_pa)

        print(vv_ua)

def group_by_aspect_individual_orbits(in_validated):
    df_ref = read_reference(in_reference, in_dmr)
    df_val = read_validated(in_validated)
    df = df_ref.merge(df_val, left_index=True, right_index=True)

    out_list = []
    dfs = [
        df[(df['ASPECT_mean'] <= 45) | (df['ASPECT_mean'] > 315)],
        df[(df['ASPECT_mean'] > 45) & (df['ASPECT_mean'] <= 135)],
        df[(df['ASPECT_mean'] > 135) & (df['ASPECT_mean'] <= 225)],
        df[(df['ASPECT_mean'] > 225) & (df['ASPECT_mean'] <= 315)]
    ]

    for df_filter, aspect in zip(dfs, ('N', 'E', 'S', 'W')):
        tp = df_filter['TP'].sum()
        fp = df_filter['FP'].sum()
        fn = df_filter['FN'].sum()

        ua = tp / (tp + fp)
        pa = tp / (tp + fn)

        f1 = 2 * (ua * pa) / (ua + pa)
        print(in_validated[-10:-8], in_validated[-7:-4], aspect)
        print('UA, PA, F1')
        print(ua, pa, f1)
    print('---------------------------')


def group_by_incidence_angle_individual_orbits(in_validated, overall, incidence_ranges):
    ro = int(in_validated[-7:-4])

    df_ref = read_reference(in_reference, in_dmr)
    df_val = read_validated(in_validated)
    df = df_ref.merge(df_val, left_index=True, right_index=True)

    dfs = [
        df[(df[f'RO{ro}_mean'] <= 25) & (df[f'RO{ro}_mean'] > 15)],
        df[(df[f'RO{ro}_mean'] <= 30) & (df[f'RO{ro}_mean'] > 25)],
        df[(df[f'RO{ro}_mean'] <= 35) & (df[f'RO{ro}_mean'] > 30)],
        df[(df[f'RO{ro}_mean'] <= 40) & (df[f'RO{ro}_mean'] > 35)],
        df[(df[f'RO{ro}_mean'] <= 45) & (df[f'RO{ro}_mean'] > 40)],
        df[(df[f'RO{ro}_mean'] <= 50) & (df[f'RO{ro}_mean'] > 45)],
        df[(df[f'RO{ro}_mean'] <= 60) & (df[f'RO{ro}_mean'] > 50)],
    ]

    for df_filter, incidence_angle in zip(dfs, incidence_ranges):
        tp = df_filter['TP'].sum()
        fp = df_filter['FP'].sum()
        fn = df_filter['FN'].sum()

        overall[incidence_angle]['TP'] += tp
        overall[incidence_angle]['FP'] += fp
        overall[incidence_angle]['FN'] += fn

        ua = tp / (tp + fp)
        pa = tp / (tp + fn)

        f1 = 2 * (ua * pa) / (ua + pa)
        #print(in_validated[-10:-8], in_validated[-7:-4], incidence_angle)
        #print(tp, fp, fn)
        #print('UA, PA, F1')
        #print(ua, pa, f1)
    #print('---------------------------')
    return overall

def print_overall(x, incidence_ranges):
    for i in incidence_ranges:
        tp = x[i]['TP']
        fp = x[i]['FP']
        fn = x[i]['FN']

        ua = tp / (tp + fp)
        pa = tp / (tp + fn)

        f1 = 2 * (ua * pa) / (ua + pa)
        #print(i)
        #print('UA, PA, F1')
        #print(ua, pa, f1)
    #print('---------------------------')

if __name__ == '__main__':
    in_reference = r'C:\Users\dd\Documents\NATUR_CUNI\_dp\reference_data\LocalIncidenceAngles_ZonalStats_5514_Area.gpkg'
    in_dmr = r'C:\Users\dd\Documents\NATUR_CUNI\_dp\reference_data\DTM_ZonalStats.gpkg'
    in_validated = r'C:\Users\dd\Documents\NATUR_CUNI\_dp\results\Results_validated_2021.csv'
    in_validated = r'C:\Users\dd\Documents\NATUR_CUNI\_dp\results\validated_2021.csv'
    in_validated_py = glob(r'C:\Users\dd\Documents\NATUR_CUNI\_dp\results\validated_py_*')
    print(in_validated_py)
    df_ref = read_reference(in_reference, in_dmr)
    df_val = read_validated(in_validated)
    df_merge = df_ref.merge(df_val, left_index=True, right_index=True)
    print(df_merge.columns)

    #plot_metric(df_merge, y_lab='F1')
    #group_by_area_new(df_merge, level='5')
    #group_by_aspect(df_merge)

    incidence_ranges = ('<25', '<30', '<35', '<40', '<45', '<50', '<60')
    overall = {}
    for i in incidence_ranges:
        overall[i] = {'TP': 0, 'FP': 0, 'FN': 0}

        for validated_py in in_validated_py:
        #group_by_aspect_individual_orbits(validated_py)

            overall = group_by_incidence_angle_individual_orbits(validated_py, overall, incidence_ranges)
    print(overall)
    print_overall(overall, incidence_ranges)

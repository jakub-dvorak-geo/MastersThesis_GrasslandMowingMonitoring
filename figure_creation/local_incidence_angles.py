import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np


def read_angles(in_path, metric='mean'):
    in_gdf = gpd.read_file(in_path)
    gdf = in_gdf.filter(regex=f'.*_{metric}', axis='columns')
    return gdf

def create_violin_plot(gdf):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,
        figsize=(9, 4), sharey=True, layout='tight')
    ax1.grid(axis='y', color='grey', linestyle='--')
    ax1.grid(axis='x', color='grey', linestyle='--', which='minor')
    ax1.violinplot(gdf.loc[:,['RO22_mean', 'RO95_mean']])
    ax1.set_xticks([1, 2], labels=['22', '95'], minor=False, fontsize=12)
    ax1.set_xticks([0.75, 1.25, 1.75, 2.25], labels=['0.25', '0.25', '0.25', '0.25'], minor=True, fontstyle='italic')
    ax1.set_xlabel('Descending')
    ax1.set_ylabel('Local Incidence Angle [°]')
    ax1.yaxis.tick_right()
    ax1.yaxis.set_tick_params(pad=12)

    ax2.grid(axis='y', color='grey', linestyle='--')
    ax2.grid(axis='x', color='grey', linestyle='--', which='minor')
    ax2.violinplot(gdf.loc[:,['RO73_mean', 'RO146_mean']])
    ax2.set_xticks([1, 2], labels=['73', '146'], minor=False, fontsize=12)
    ax2.set_xticks([0.75, 1.25, 1.75, 2.25], labels=['0.25', '0.25', '0.25', '0.25'], minor=True, fontstyle='italic')
    ax2.set_xlabel('Ascending')
    plt.show()

if __name__ == '__main__':
    in_gpkg = r'C:\Users\dd\Documents\NATUR_CUNI\_dp\reference_data\LocalIncidenceAngles_ZonalStats.gpkg'

    gdf_means = read_angles(in_gpkg, 'mean')
    print(gdf_means)
    create_violin_plot(gdf_means)

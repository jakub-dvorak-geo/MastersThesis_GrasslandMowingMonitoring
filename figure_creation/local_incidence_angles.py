import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np


def read_angles(in_path, metric='mean'):
    in_gdf = gpd.read_file(in_path)
    gdf = in_gdf.filter(regex=f'.*_{metric}', axis='columns')
    return gdf

def create_violin_plot(gdf):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,
        figsize=(9, 4), sharey=True)
    ax1.violinplot(gdf.loc[:,['RO22_mean', 'RO95_mean']])
    ax1.set_xticks([1, 2], labels=['22', '95'])
    ax1.set_xlabel('Descending')
    ax1.yaxis.tick_right()
    #ax1.set_ylabel('Observed values')
    ax2.violinplot(gdf.loc[:,['RO73_mean', 'RO146_mean']])
    ax2.set_xticks([1, 2], labels=['73', '146'])
    ax2.set_xlabel('Ascending')
    ax2.set_ylabel('Local Incidence Angle [°]')
    plt.show()

if __name__ == '__main__':
    in_gpkg = r'C:\Users\dd\Documents\NATUR_CUNI\_dp\reference_data\LocalIncidenceAngles_ZonalStats.gpkg'

    gdf_means = read_angles(in_gpkg, 'mean')
    print(gdf_means)
    create_violin_plot(gdf_means)

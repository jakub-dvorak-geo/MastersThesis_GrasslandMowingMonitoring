
import geopandas as gpd
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import sys
import json
from os.path import join


def read_data(in_path):
    gpkg = gpd.read_file(in_path)
    return {'aspect_mean': gpkg['ASPECT_mean']/180*np.pi,
        'slope_mean': gpkg['SLOPE_mean'],
        'aspect_std': gpkg['ASPECT_stdev']/180*np.pi,
        'slope_std': gpkg['SLOPE_stdev'],}


def create_plot(data):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='polar', theta_offset=np.pi/2)
    ax.plot(data['aspect_mean'], data['slope_mean'], 'o')
    #ax.errorbar(data['aspect_mean'], data['slope_mean'],
    #    data['slope_std'], data['aspect_std'],'o')
    ax.arrow(0, 0, 0, 25, length_includes_head=True,
        head_width=0.04, head_length=1)
    ax.set_rlabel_position(0.15)
    ax.set_rlim(0, 25)
    theta_labels = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    ax.set_theta_direction('clockwise')
    ax.set_thetagrids([0,45,90,135,180,225,270,315], labels=theta_labels)
    fig.suptitle('Topography of monitored plots')
    #ax.set_title('Slope and Exposure')
    ax.set_xlabel('Average exposure')
    ax.text(-0.1, 12.5, 'Average slope [%]', rotation='vertical')
    plt.show()


if __name__ == '__main__':
    data_path = r'C:\Users\dd\Documents\NATUR_CUNI\_dp\reference_data\DTM_ZonalStats.gpkg'
    loaded_data = read_data(data_path)
    create_plot(loaded_data)

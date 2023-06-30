import geopandas as gpd
import matplotlib.pyplot as plt
from pandas import to_datetime

from os.path import join


root_dir = r'C:\Users\dd\Documents\NATUR_CUNI\_dp\reference_data'

file_2021 = join(root_dir, r'Vyjezdy_2021\slc_products_2021.gpkg')
file_2022 = join(root_dir, r'Vyjezdy_2022\slc_products_2022.gpkg')

def plot_year(year, file, axis=None):
    gpkg = gpd.read_file(file)
    gpkg['beginposition'] = to_datetime(gpkg['beginposition'])

    axis.set_title(str(year), loc='left')
    axis.set_ylim((0,1))
    axis.tick_params(left = False, right = False , labelleft = False)
    colors = {
        22: 'red',
        73: 'green',
        95: 'blue',
        146:'black'
    }

    for ro in gpkg['relativeorbitnumber'].unique():
        gpkg_filter = gpkg.loc[gpkg['relativeorbitnumber'] == ro]
        axis.eventplot(gpkg_filter['beginposition'], orientation="horizontal",
            linewidth=1.5, lineoffsets=0.5, colors=colors[ro])


def combine_subplots():
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, layout='constrained')
    fig.suptitle('Sentinel-1 observation dates')

    plot_year(2021, file_2021, ax1)
    fig.legend(loc='outside lower center', title='legend')
    plot_year(2022, file_2022, ax3)
    plt.show()

combine_subplots()



from os import listdir
from os.path import join
from glob import glob
import sys
import json

def get_filenames(in_dir):

    out_dict = {}
    for year in glob(join(in_dir, '[0-9][0-9][0-9][0-9]')):
        out_dict[year] = {}
        for ro in listdir(join(in_dir, year)):
            out_dict[year][ro] = listdir(join(in_dir, year, ro))
    print(out_dict)
    return out_dict

def extract_dates(in_dict):
    out_dict = {}
    for year in in_dict.keys():
        out_dict[year] = {}
        for ro in in_dict[year].keys():
            out_dict[year][ro] = [i[17:25] for i in in_dict[year][ro]]
    print(out_dict)
    return out_dict

def save_json(in_dict, out_path):
    with open(out_path, 'w') as out_file:
        json.dump(in_dict, out_file)

if __name__ == '__main__':

    if len(sys.argv) > 1:
        root_dir = sys.argv[1]
    else:
        root_dir = 'f:/datasets/NATUR_CUNI/s1_download'
    out_path = join(root_dir, 'obs_dates.json')

    filenames_dict = get_filenames(root_dir)
    dates_dict = extract_dates(filenames_dict)
    save_json(dates_dict, out_path)

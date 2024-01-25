from constants import DATA_PATH, EXP_PATH, data_list
import time 
import utils_math
import scipy.io as sio
import logging 
import os 
from datetime import datetime
import argparse 
from algos import closest_to_cluster_centroids_approach_v2
import pandas as pd 


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=0, type=int, help="Starting index of data list")
    parser.add_argument("--end", default=len(data_list), type=int, help="Ending index of data list")
    args = parser.parse_args()


    columns = ['C: Contacts', 'M: polys', 'D: Measurements', 'V^(1/D)', 'Indices']

    for angle in [5, 10, 15]: 
        df = []
        print(f"Running for angle threshold: {angle}")
        for mat_info in data_list[args.start:args.end]: 
            C, M, D = mat_info
            print(f"Running for: {C}contacts_{M}polys_D{D}")
            data = sio.loadmat(DATA_PATH / 'OptimizationMatrices' / f'{C}contacts_{M}polys.mat')
            arr = data['JGQ']
            # EXP_DIR = EXP_PATH / 'OptimizationMatrices' / f'{C}contacts_{M}polys_D{D}'
            # os.makedirs(EXP_DIR, exist_ok=True)

            arr = utils_math.scale_data(arr, technique='simple')
            vol, selected_indices = closest_to_cluster_centroids_approach_v2(arr, num_entries=D, angle_threshold=angle)
            df.append([C, M, D, vol, selected_indices])

        df = pd.DataFrame(df, columns=columns)
        df.to_csv(EXP_PATH / 'OptimizationMatrices' / f'clustering_ang{angle}_mag.csv', index=False) 

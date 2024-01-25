from constants import DATA_PATH, EXP_PATH, data_list
import time 
import utils_math
import scipy.io as sio
import logging 
import os 
from datetime import datetime
import argparse 
import algos 
import pandas as pd 


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=0, type=int, help="Starting index of data list")
    parser.add_argument("--end", default=len(data_list), type=int, help="Ending index of data list")
    args = parser.parse_args()


    df_angle = []
    df_mag = []
    df_rad = []

    columns = ['C: Contacts', 'M: polys', 'D: Measurements', 'V^(1/D)', 'Indices']
    # Load the data 
    for mat_info in data_list[args.start:args.end]: 
        C, M, D = mat_info
        print(f"Running for: {C}contacts_{M}polys_D{D}")
        data = sio.loadmat(DATA_PATH / 'OptimizationMatrices' / f'{C}contacts_{M}polys.mat')
        arr = data['JGQ']
        # EXP_DIR = EXP_PATH / 'OptimizationMatrices' / f'{C}contacts_{M}polys_D{D}'
        # os.makedirs(EXP_DIR, exist_ok=True)
        arr = utils_math.scale_data(arr, technique='simple')

        vol1, selected_indices1 = algos.closest_to_cluster_centroids_approach(arr, num_entries=D) 
        df_angle.append([C, M, D, vol1, selected_indices1])

        vol2, selected_indices2 = algos.highest_mag_approach(arr, num_entries=D)
        df_mag.append([C, M, D, vol2, selected_indices2])

        vol3, selected_indices3, _, _, _, _ = algos.max_radial_coors_approach(arr, 
                                                                  clust_param_dict={'n_clusters': D, 
                                                                                    'n_init': 'auto', 
                                                                                    'random_state': 0})
        df_rad.append([C, M, D, vol3, selected_indices3])

    df_angle = pd.DataFrame(df_angle, columns=columns)
    df_angle.to_csv(EXP_PATH / 'OptimizationMatrices' / 'clustering_angle.csv', index=False)

    df_mag = pd.DataFrame(df_mag, columns=columns)
    df_mag.to_csv(EXP_PATH / 'OptimizationMatrices' / 'clustering_mag.csv', index=False)

    df_rad = pd.DataFrame(df_rad, columns=columns)
    df_rad.to_csv(EXP_PATH / 'OptimizationMatrices' / 'clustering_rad.csv', index=False)



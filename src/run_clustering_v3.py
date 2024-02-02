from constants import DATA_PATH, EXP_PATH, data_list
import time 
import utils_math
import scipy.io as sio
import logging 
import os 
from datetime import datetime
import argparse 
from algos import closest_to_cluster_centroids_approach_v3
import pandas as pd 
import time 


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=0, type=int, help="Starting index of data list")
    parser.add_argument("--end", default=len(data_list), type=int, help="Ending index of data list")
    parser.add_argument("--angles", nargs="+", required=True, help="Angle Threshold")
    parser.add_argument("--num_mags", type=int, required=True, 
                        help="Number of highest ranked mags to select as candidates")
    args = parser.parse_args()

    args.angles = list(map(float, args.angles))
    columns = ['C: Contacts', 'M: polys', 'D: Measurements', 'V^(1/D)', 'Indices', 'Elasped Time']

    for angle in args.angles: 
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
            start = time.time()
            vol, selected_indices = closest_to_cluster_centroids_approach_v3(arr, num_entries=D, angle_threshold=angle,
                                                                             num_mags=args.num_mags)
            elapsed_time = time.time() - start 

            df.append([C, M, D, vol, selected_indices, elapsed_time])

        df = pd.DataFrame(df, columns=columns)
        df.to_csv(EXP_PATH / 'OptimizationMatrices' / f'clustering_ang{int(angle)}_{args.num_mags}mag.csv', index=False) 
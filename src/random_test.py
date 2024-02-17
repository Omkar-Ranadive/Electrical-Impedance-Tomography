import utils_math, utils_vis
from constants import DATA_PATH, EXP_PATH, data_list
import scipy.io as sio
import numpy as np 
import algos 
from algo_genetic import Genetic
import logging 
import argparse 
import os 
from datetime import datetime
import time 
from math import comb 


def print_results(vol, indices, rank_indices, algo_info): 
    logger.info(f"Max volume: {vol}")
    logger.info(f"Selected Indices: {indices}")
    logger.info("Rank of selected indices: ")
    logger.info(rank_indices)
    cart_dict, cart_counts = utils_math.get_cluster_for_index(clust_labels_cartesian, indices)
    ang_dict, ang_counts = utils_math.get_cluster_for_index(clust_labels_angular, indices)
    logger.info(f"Cartesian Clustering Labels: {cart_dict}")
    logger.info(f"Cartesian Clustering Counts: {cart_counts}")

    logger.info(f"Angular Clustering Labels: {ang_dict}")
    logger.info(f"Angular Clustering Counts: {ang_counts}")

    utils_vis.plot_clust_index_dist(cart_counts, ang_counts, 
                                    filename=EXP_DIR / f'clust_dist_{algo_info}.png', algo_info=algo_info)
    logger.info("*"*20) 


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", required=True, type=str)
    args = parser.parse_args()

    EXP_DIR = EXP_PATH / args.exp_name 
    os.makedirs(EXP_DIR, exist_ok=True)

    # Set up logger 
    logging.basicConfig(level=logging.INFO, filename=str(EXP_DIR / 'info.log'), format='%(message)s')
    logger=logging.getLogger() 
    time_stamp = datetime.now()
    logger.info(f"Running random_test.py: {time_stamp.strftime('%H:%M:%S')}") 

    # Load the data 
    # data = sio.loadmat(DATA_PATH / '9ContactsMatrix.mat')
    for i in range(5): 
        C, M, D = data_list[i] 
        rows = 3*comb(C, 4)
        
        arr = np.random.randn(rows, M)
        logger.info("*"*20) 
        logger.info(f"Shape of input array: {arr.shape}")

        # # Scale/normalize the data 
        # arr = utils_math.scale_data(arr, technique='simple')

        # Hyperparameters 
        seed = 0 
        num_entries = D
        logger.info(f"Measurement (K) (num entries): {num_entries}")
        logger.info(f"Random seed used: {seed}")

        # Cluster the data in cartesian coordinates 
        param_dict = {'n_clusters': num_entries, 'n_init': 'auto', 'random_state': seed}
        clust_labels_cartesian, _, _ = utils_math.get_kmeans_clusters(arr, param_dict) 

        # Cluster the data in spherical coordiantes 
        param_dict = {'n_clusters': num_entries, 'n_init': 'auto', 'random_state': seed}
        max_rad_vol, max_rad_indices, clust_labels_angular, _, _, _ = algos.max_radial_coors_approach(arr, param_dict) 


        magnitudes, sorted_indices = utils_math.get_norm_with_rank(arr) 
        logger.info("Indices ranked by magnitude")
        logger.info(sorted_indices)
        logger.info("*"*20)

        logger.info("Highest magnitudes approach: ")
        mag_vol = utils_math.cal_vol(arr[sorted_indices[:num_entries]])
        mag_indices = sorted_indices[:num_entries]
        mag_ranks = utils_math.get_indices_rank(sorted_indices, mag_indices)
        print_results(mag_vol, mag_indices, mag_ranks, algo_info="highest_mag")
        logger.info("*"*20)

        logger.info("Greedy approach: ") 
        greedy_vol, greedy_indices = algos.greedy_approach(arr, num_entries=num_entries) 
        greedy_mag_ranks = utils_math.get_indices_rank(sorted_indices, greedy_indices)
        print_results(greedy_vol, greedy_indices, greedy_mag_ranks, algo_info="greedy")

        logger.info("Max radial coordinate approach: ")
        max_rad_ranks = utils_math.get_indices_rank(sorted_indices, max_rad_indices)
        print_results(max_rad_vol, max_rad_indices, max_rad_ranks, algo_info="max_radial_coordinate")

        logger.info("Max radial coordinate approach with flipped signs: ")
        arr_flipped = utils_math.flip_signs(arr)
        max_rad_vol_flip, max_rad_indices_flip, clust_labels_flip, _, _, _ = algos.max_radial_coors_approach(
                                                                                                    arr_flipped, 
                                                                                                    param_dict) 
        max_rad_flip_ranks = utils_math.get_indices_rank(sorted_indices, max_rad_indices_flip)
        print_results(max_rad_vol_flip, max_rad_indices_flip, max_rad_flip_ranks, algo_info="max_radial_flipped")

        logger.info("Max radial coordinates with each axis as init centroid")
        initial_centers = np.eye(N=num_entries, M=arr.shape[1]-1)
        param_dict = {'n_clusters': num_entries, 'n_init': 'auto', 'random_state': seed, 'init': initial_centers}
        max_rad_vol_axis, max_rad_indices_axis, clust_labels_axis, _, _, _ = algos.max_radial_coors_approach(arr, 
                                                                                                             param_dict) 
        max_rad_axis_ranks = utils_math.get_indices_rank(sorted_indices, max_rad_indices_axis)
        print_results(max_rad_vol_axis, max_rad_indices_axis, max_rad_axis_ranks, algo_info="max_radial_axis_centroid")

        logger.info("Random volume approach (avg over 1000 iterations)")
        avg_ran_vol, best_ran_vol = algos.random_approach(arr, num_entries=num_entries, num_iterations=5000) 
        logger.info(f"Avg random volume: {avg_ran_vol}. Best random volume: {best_ran_vol}")
        logger.info("*"*20)

        logger.info("Genetic algorithm approach")
        gen_dict = {'arr': arr, 'num_generations': 500, 'population_size': 5000, 'num_parents': 200, 
                    'mutation_rate': 0.2, 'num_entries': num_entries, 'adaptive_mutation': False, 
                    'improvement_check': 50, 'random_children_rate': 0.3, 
                    'direct_mutants_rate': 0.05, 'print_every': 100}
        logger.info(gen_dict)
        genObj = Genetic(**gen_dict)
        best_results = genObj.run_genetic_simulation() 
        genetic_indices_ranks = utils_math.get_indices_rank(sorted_indices, best_results['best_indices'])
        logger.info(f"Best Generation: {best_results['best_generation']}")
        print_results(best_results['best_score'], best_results['best_indices'], genetic_indices_ranks, algo_info="genetic")

        logger.info("Modified Cosine: ") 
        max_cos_vol, max_cos_indices, clust_labels = algos.cosine_clustering(arr, num_entries=num_entries)
        max_cos_ranks = utils_math.get_indices_rank(sorted_indices, max_cos_indices)
        print_results(max_cos_vol, max_cos_indices, max_cos_ranks, algo_info="modified_cosine")

        logger.info("Left Singular Approach: ")
        ls_vol, ls_indices = algos.left_singular_matrix_approach(arr, num_entries) 
        ls_ranks = utils_math.get_indices_rank(ls_indices, ls_indices)
        print_results(ls_vol, ls_indices, ls_ranks, algo_info="left_singular_vals")

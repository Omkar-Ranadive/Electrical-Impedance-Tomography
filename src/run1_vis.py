from constants import DATA_PATH, EXP_PATH
import time 
import utils_math
import utils_vis
import scipy.io as sio
import logging 
import os 
from datetime import datetime
import pandas as pd 
from algos import max_radial_coors_approach
import numpy as np 
import ast
import sys 
import logging 


if __name__ == '__main__': 
    df = pd.read_csv(EXP_PATH / 'OptimizationMatrices' / 'volumes.csv')    
    print(df.columns)
    volumes1 = []
    volumes2 = []

    for index, row in df.iterrows():         
        C, M, D = row['C: Contacts'], row['M: polys'], row['D: Measurements']
        print(f"Running for: {C}contacts_{M}polys_D{D}")
        data = sio.loadmat(DATA_PATH / 'OptimizationMatrices' / f'{C}contacts_{M}polys.mat')
        arr = data['JGQ']
        print(arr.shape)
        EXP_DIR = EXP_PATH / 'OptimizationMatrices' / f'{C}contacts_{M}polys_D{D}'
        os.makedirs(EXP_DIR, exist_ok=True)
        num_entries = D
        seed = 0

        logging.basicConfig(level=logging.INFO, filename=str(EXP_DIR / 'info_vis.log'), 
                            format='%(message)s', filemode='w')
        logger=logging.getLogger() 
        logging.info("*"*20)
        logging.info(f"{C}contacts_{M}polys_D{D}")
        logging.info(f"Shape of array: {arr.shape}")
        # Get magnitudes and indices sorted by magnitude 
        magnitudes, sorted_mag_indices = utils_math.get_norm_with_rank(arr) 

        # Cluster the data in cartesian coordinates 
        param_dict = {'n_clusters': num_entries, 'n_init': 'auto', 'random_state': seed}
        clust_labels_cartesian, clust_centers, clust_dist = utils_math.get_kmeans_clusters(arr, param_dict) 
        logger.info("Cartesian Clustering distribution")
        logger.info(clust_dist)
      
        # Cluster the data in spherical coordiantes 
        param_dict = {'n_clusters': num_entries, 'n_init': 'auto', 'random_state': seed}
        max_rad_vol, max_rad_indices, clust_labels_angular, ang_centers, thetas, ang_dist = max_radial_coors_approach(
                                                                                                        arr, 
                                                                                                        param_dict)
        logger.info("Angular clustering distribution:")
        logger.info(ang_dist)

        # Plot the cluster distributions 
        utils_vis.plot_cluster_distribution(clust_dist, title='Cartesian', 
                                            filename=EXP_DIR / f'{C}contacts_{M}polys_D{D}_cartesian_full_dist.png')
        
        utils_vis.plot_cluster_distribution(ang_dist, title='Angular', 
                                            filename=EXP_DIR / f'{C}contacts_{M}polys_D{D}_angular_full_dist.png')

        # Get the volume information 
        vol1 = row['V^(1/D)']*10
        vol2 = row['V^(1/D) #2']
        volumes1.append(vol1) 
        volumes2.append(vol2)
        indices = np.array(ast.literal_eval(row['Indices #2']), dtype=int)  # Convert string to numpy array 

        # Plot number of selected indices belonging to each cluster 
        cart_dict, cart_counts = utils_math.get_cluster_for_index(clust_labels_cartesian, indices)
        ang_dict, ang_counts = utils_math.get_cluster_for_index(clust_labels_angular, indices)

        utils_vis.plot_clust_index_dist(cart_counts, ang_counts, 
                                    filename=EXP_DIR / f'clust_dist_ensembleGen.png', algo_info='ensembleGen')

        # Get distribution of angles w.r.t cluster centers and distribution of magnitudes within clusters 
        if num_entries <= 20: 
            cartesian_angles = utils_math.get_angles_per_cluster(arr, clust_labels_cartesian, clust_centers)
            cartesian_indices_angles = utils_math.get_angles_per_cluster(arr[indices], clust_labels_cartesian[indices], 
                                                                         clust_centers)
            
            utils_vis.plot_angle_distribution(cartesian_angles, indices_dict=cartesian_indices_angles,
                                              title="Cartesian Clustering",
                                            filename=EXP_DIR / f'{C}contacts_{M}polys_D{D}_cartesian_angle_dist.png')
            


            angular_angles = utils_math.get_angles_per_cluster(thetas, clust_labels_angular, ang_centers)
            angular_indices_angles = utils_math.get_angles_per_cluster(thetas[indices], clust_labels_angular[indices], 
                                                                       ang_centers)
            
            utils_vis.plot_angle_distribution(angular_angles, indices_dict=angular_indices_angles,
                                              title="Angular Clustering",
                                              filename=EXP_DIR / f'{C}contacts_{M}polys_D{D}_angular_angle_dist.png')
            

            cartesian_mags = utils_math.get_mags_per_cluster(arr, clust_labels_cartesian) 
            cartesian_indices_mags = utils_math.get_mags_per_cluster(arr[indices], clust_labels_cartesian[indices])
            utils_vis.plot_mag_distribution(cartesian_mags, cartesian_indices_mags, title="Cartesian Clustering", 
                                            filename=EXP_DIR / f'{C}contacts_{M}polys_D{D}_cartesian_mag_dist.png')


        # Get angles between the cluster centers 
        cart_centroid_angles = utils_math.get_angles_between_centroids(clust_centers)
        ang_centroid_angles = utils_math.get_angles_between_centroids(ang_centers) 
        utils_vis.plot_centroid_angle_heatmap(cart_centroid_angles, title="Cartesian",
                                              filename=EXP_DIR / (f'{C}contacts_{M}polys_D{D}_cartesian'
                                                                  f'_centroids_angles.png'))
        utils_vis.plot_centroid_angle_heatmap(ang_centroid_angles, title="Angular",
                                              filename=EXP_DIR / (f'{C}contacts_{M}polys_D{D}_angular'
                                                                  f'_centroids_angles.png'))

        # Plot the ranks (based on magnitude) for the selected indicies 
        indices_ranks = utils_math.get_indices_rank(sorted_mag_indices, indices)
        logging.info("Magnitude ranks of selected indices")
        logging.info(indices_ranks)
        utils_vis.plot_magnitude_ranks(magnitudes, sorted_mag_indices, indices_ranks, 
                                       filename=EXP_DIR / f'{C}contacts_{M}polys_D{D}_mag_ranks.png')
        
        logger.handlers.clear()


    utils_vis.compare_volumes(volumes1, volumes2, filename=EXP_PATH / 'OptimizationMatrices' / 'volumes_compare.png', 
                              algo_info='Comparison between volumes')


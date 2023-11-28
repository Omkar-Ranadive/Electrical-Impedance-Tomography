import numpy as np 
import utils 
from constants import DATA_PATH 
import scipy.io as sio
from itertools import combinations
from sklearn.cluster import DBSCAN
import cosine_kmeans


def greedy_approach(arr, num_entries):
    """
    Find the subset of k entries (k=num_entries) which (tries) to maximizes the volume using greedy approach. 
    At each step, we choose the best possible entry which maximizes the volume. 
    Note, as this is a greedy approach, the solution is not optimal. 

    Args:
        arr (np.ndarray): Input array 
        num_entries (int): Subset to be used to find max volume.

    Returns: 
        final_volume (float): Max volume obtained 
        selected_indice (list): Indices selected to calculate the volume 
    """

    total_indices = np.arange(0, arr.shape[0])
    selected_indices = []
    for i in range(num_entries): 
        remaining_indices = np.setdiff1d(total_indices, selected_indices)

        volumes = []
        for index in remaining_indices: 
            arr_subset = arr[selected_indices + [index], :]
            volumes.append(utils.cal_vol(arr_subset))
        
        max_vol_index = np.argmax(volumes) 
        selected_indices.append(remaining_indices[max_vol_index])


    arr_subset = arr[selected_indices, :]
    final_volume = utils.cal_vol(arr_subset) 

    return final_volume, sorted(selected_indices) 


def max_radial_coors_approach(arr, clust_param_dict): 
    """
    Convert the input data in cartesian coordinates to nsphere coordinates. Cluster the angular coordinates, 
    take row corresponding to the max radial coordinate of each cluster and use it for volumne calculation. 

    Args:
        arr (np.ndarray): Input array 
        clust_param_dict: Dictionary containing k-means hyperparamters. 
            n_clusters (int, required): Number of clusters (= num_entries)
            random_state (int, optional): If provided, clustering is done with a fixed a random seed  
            n_init (str, optional): Initialization strategy. (Recommended='auto')
            init (arr, optional): Specify intial centroids if required 
    Returns:
        final_volume (float): Max volume obtained 
        selected_indice (list): Indices selected to calculate the volume     
    
    """

    max_per_cluster = [] 
    # Convert cartesian to n-sphere coordinates 
    r, thetas = utils.cartesian_to_nsphere(arr) 
    # print("Shape of angular coordinates", thetas.shape)

    # Cluster the angular coordinates (thetas) 
    labels, clust_dist = utils.get_kmeans_clusters(thetas, clust_param_dict)
    # print("Cluster distribution: ", clust_dist)

    # Get radial coordinates for each point in each cluster 
    r_dict = utils.radial_coord_per_cluster(arr, labels) 
    
    # Get max radial coordinate from each cluster
    selected_indices = []
    for label, rvalues in r_dict.items():
        r_coors, indices = rvalues[0], rvalues[1] 
        max_per_cluster.append(np.max(r_coors))
        max_index = np.argmax(r_coors)
        selected_indices.append(indices[max_index])

    # Calculate volume based on these radial coordinates 
    final_volume = utils.cal_vol(arr[selected_indices, :]) 

    return final_volume, sorted(selected_indices), labels


def random_approach(arr, num_entries, num_iterations=1000): 

    volumes = []
    total_values = len(arr) 

    for _ in range(num_iterations): 
        random_indices = np.random.choice(total_values, num_entries, replace=False)
        arr_subset = arr[random_indices, :]
        volumes.append(utils.cal_vol(arr_subset))

    avg_vol = np.mean(volumes) 

    return avg_vol


def cosine_clustering(arr): 
    ck_obj = cosine_kmeans.CosineKMeans(n_clusters=26, max_iterations=50000)
    ck_obj.fit(arr) 
    labels = ck_obj.predict(arr) 
    unique, counts = np.unique(labels, return_counts=True)
    clust_dist = dict(zip(unique, counts))
    print(clust_dist)
    # Get radial coordinates for each point in each cluster 
    r_dict = utils.radial_coord_per_cluster(arr, labels) 
    max_per_cluster = [] 

    # Get max radial coordinate from each cluster
    selected_indices = []
    for label, rvalues in r_dict.items():
        r_coors, indices = rvalues[0], rvalues[1] 
        max_per_cluster.append(np.max(r_coors))
        max_index = np.argmax(r_coors)
        selected_indices.append(indices[max_index])

    # Calculate volume based on these radial coordinates 
    final_volume = utils.cal_vol(arr[selected_indices, :]) 

    return final_volume, sorted(selected_indices), labels




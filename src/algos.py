import numpy as np 
import utils_math 
from constants import DATA_PATH 
import scipy.io as sio
from itertools import combinations
from sklearn.cluster import DBSCAN
import custom_kmeans
from scipy import linalg 


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
            volumes.append(utils_math.cal_vol(arr_subset))
        
        max_vol_index = np.argmax(volumes) 
        selected_indices.append(remaining_indices[max_vol_index])


    arr_subset = arr[selected_indices, :]
    final_volume = utils_math.cal_vol(arr_subset) 

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
        labels (np.ndarray): Cluster labels 
        cluster_center (np.ndarray): Cluster centers
    """

    max_per_cluster = [] 
    # Convert cartesian to n-sphere coordinates 
    r, thetas = utils_math.cartesian_to_nsphere(arr) 
    # print("Shape of angular coordinates", thetas.shape)

    # Cluster the angular coordinates (thetas) 
    labels, clust_centers, clust_dist = utils_math.get_kmeans_clusters(thetas, clust_param_dict)
    # print("Cluster distribution: ", clust_dist)

    # Get radial coordinates for each point in each cluster 
    r_dict = utils_math.radial_coord_per_cluster(arr, labels) 
    
    # Get max radial coordinate from each cluster
    max_per_cluster, selected_indices = utils_math.max_radial_coordinate_per_cluster(r_dict)

    # Calculate volume based on these radial coordinates 
    final_volume = utils_math.cal_vol(arr[selected_indices, :]) 

    return final_volume, sorted(selected_indices), labels, clust_centers, thetas


def random_approach(arr, num_entries, num_iterations=1000): 

    volumes = []
    total_values = len(arr) 
    
    for _ in range(num_iterations): 
        random_indices = np.random.choice(total_values, num_entries, replace=False)
        arr_subset = arr[random_indices, :]
        volumes.append(utils_math.cal_vol(arr_subset))

    avg_vol = np.mean(volumes) 
    best_vol = np.max(volumes)

    return avg_vol, best_vol


def cosine_clustering(arr, num_entries): 
    ck_obj = custom_kmeans.CosineKMeans(n_clusters=num_entries, max_iterations=50000)
    ck_obj.fit(arr) 
    labels = ck_obj.predict(arr) 
    unique, counts = np.unique(labels, return_counts=True)
    clust_dist = dict(zip(unique, counts))
    print(clust_dist)
    # Get radial coordinates for each point in each cluster 
    r_dict = utils_math.radial_coord_per_cluster(arr, labels)
     
    # Get max radial coordinate from each cluster
    max_per_cluster, selected_indices = utils_math.max_radial_coordinate_per_cluster(r_dict)

    # Calculate volume based on these radial coordinates 
    final_volume = utils_math.cal_vol(arr[selected_indices, :]) 

    return final_volume, sorted(selected_indices), labels


def left_singular_matrix_approach(arr, num_entries): 
    U, s, Vh = linalg.svd(arr)
    avg_singular_values = np.mean(U, axis=1)
    selected_indices = np.argsort(avg_singular_values)[-num_entries:]
    final_volume = utils_math.cal_vol(arr[selected_indices, :])

    return final_volume, sorted(selected_indices)


def ortho_kmeans(arr, num_entries):
    kmeans_ortho = custom_kmeans.OrthogonalKMeans(n_clusters=num_entries, regularization_strength=0.5)
    kmeans_ortho.fit(arr)
    labels = kmeans_ortho.labels_
    unique, counts = np.unique(labels, return_counts=True)
    clust_dist = dict(zip(unique, counts))

    # Get radial coordinates for each point in each cluster 
    r_dict = utils_math.radial_coord_per_cluster(arr, labels) 
    max_per_cluster, selected_indices = utils_math.max_radial_coordinate_per_cluster(r_dict)
    
    # Calculate volume based on these radial coordinates 
    final_volume = utils_math.cal_vol(arr[selected_indices, :]) 
    
    return final_volume, sorted(selected_indices)



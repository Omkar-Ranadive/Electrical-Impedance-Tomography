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
        thetas (np.ndarray): Angular coordinates 
        clust_dist (dict): Distribution of points in the cluster 
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

    return final_volume, sorted(selected_indices), labels, clust_centers, thetas, clust_dist


def random_approach(arr, num_entries, num_iterations=1000): 
    """
    Randomly select indices and average over num_iterations. Useful for baselines. 
    Args:
        arr (np.ndrray): Input array 
        num_entries (int): Subset to be used to find max volume.
        num_iterations (int, optional): Number of random runs. Defaults to 1000.

    Returns:
        avg_vol (float): Average volume over all runs 
        best_vol (float): Best volumne over all runs 
    """

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
    """
    Uses modified k-means which is based on cosine distance for clustering. After clustering, the maximum magnitude 
    points from each cluster are selected.
    Args:
        arr (np.ndrray): Input array 
        num_entries (int): Subset to be used to find max volume.

    Returns:
        final_volume (float): Max volume obtained 
        selected_indice (list): Indices selected to calculate the volume    
        labels (np.ndarray): Cluster labels for each point in the array        
    """
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
    """
    The left singular value matrix (U) represents the information of the rows. This approach averages the matrix U and 
    selects the rows with highest average valeu 
    Args:
        arr (np.ndrray): Input array 
        num_entries (int): Subset to be used to find max volume.

    Returns:
        final_volume (float): Max volume obtained 
        selected_indice (list): Indices selected to calculate the volume    
    """
    U, s, Vh = linalg.svd(arr)
    avg_singular_values = np.mean(U, axis=1)
    selected_indices = np.argsort(avg_singular_values)[-num_entries:]
    final_volume = utils_math.cal_vol(arr[selected_indices, :])

    return final_volume, sorted(selected_indices)


def ortho_kmeans(arr, num_entries):
    """
    Uses modified k-means which is based on orthogonalily for clustering. After clustering, the maximum magnitude 
    points from each cluster are selected.
    Args:
        arr (np.ndrray): Input array 
        num_entries (int): Subset to be used to find max volume.

    Returns:
        final_volume (float): Max volume obtained 
        selected_indice (list): Indices selected to calculate the volume    
    """
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


def highest_mag_approach(arr, num_entries, seed=0):
    """
    Args:
        arr (np.ndrray): Input array 
        num_entries (int): Subset to be used to find max volume.
        seed (int, optional): Random seed to use while clustering . Defaults to 0.

    Returns:
        final_volume (float): Max volume obtained 
        selected_indice (list): Indices selected to calculate the volume        
    """
    
    param_dict = {'n_clusters': num_entries, 'n_init': 'auto', 'random_state': seed}
    labels, clust_centers, clust_dist = utils_math.get_kmeans_clusters(arr, param_dict)
    
    # Get magnitudes (norm) for all points in each cluster
    r_dict = utils_math.radial_coord_per_cluster(arr, labels) 
    max_per_cluster, selected_indices = utils_math.max_radial_coordinate_per_cluster(r_dict)
    
    # Calculate volume based on these magnitudes 
    final_volume = utils_math.cal_vol(arr[selected_indices, :]) 
    
    return final_volume, sorted(selected_indices)


def closest_to_cluster_centroids_approach(arr, num_entries, seed=0):
    """
    Cluster the data where num_clusters = num_entries. Select the point with minimum angle to centroid from each
    cluster. This algo is based on the idea that the centroids themselves are roughly orthogonal (+/-30) from each 
    other.
    Args:
        arr (np.ndrray): Input array 
        num_entries (int): Subset to be used to find max volume.
        seed (int, optional): Random seed to use while clustering . Defaults to 0.

    Returns:
        final_volume (float): Max volume obtained 
        selected_indice (list): Indices selected to calculate the volume        
    """

    param_dict = {'n_clusters': num_entries, 'n_init': 'auto', 'random_state': seed}
    labels, clust_centers, clust_dist = utils_math.get_kmeans_clusters(arr, param_dict)
    
    # Find the angles within a cluster for all points w.r.t centroids of each cluster 
    unique_labels = np.unique(labels)
    angles_per_label = {}
    selected_indices = []
    for label in unique_labels:
        indices = np.where(labels == label)[0]
        cluster_rows = arr[indices]
        cluster_angles = utils_math.calculate_abs_angle(cluster_rows, np.expand_dims(clust_centers[label], axis=0))
        # Get min angle 
        min_angle_index = np.argmin(cluster_angles)
        selected_indices.append(indices[min_angle_index])
        
        # angles_per_label[label] = cluster_angles
    
    final_volume = utils_math.cal_vol(arr[selected_indices, :]) 

    return final_volume, sorted(selected_indices)


def closest_to_cluster_centroids_approach_v2(arr, num_entries, angle_threshold, seed=0):
    """
    Cluster the data where num_clusters = num_entries. Select the point with minimum angle to centroid from each
    cluster. Now find all points within the range from min_angle to min_angle+angle_threshold. Then select the point 
    with highest magnitude from these points. So, this approach also gives weightage to magnitudes along with angles.  
    This algo is based on the idea that the centroids themselves are roughly orthogonal (+/-30) from each 
    Args:
        arr (np.ndrray): Input array 
        num_entries (int): Subset to be used to find max volume.
        angle_threshold (float): Angle threshold in degrees. Points within min_angle + angle_threshold are selected
        seed (int, optional): Random seed to use while clustering . Defaults to 0.

    Returns:
        final_volume (float): Max volume obtained 
        selected_indice (list): Indices selected to calculate the volume        
    """
    param_dict = {'n_clusters': num_entries, 'n_init': 'auto', 'random_state': seed}
    labels, clust_centers, clust_dist = utils_math.get_kmeans_clusters(arr, param_dict)
    
    # Find the angles within a cluster for all points w.r.t centroids of each cluster 
    unique_labels = np.unique(labels)
    angles_per_label = {}
    selected_indices = []
    for label in unique_labels:
        indices = np.where(labels == label)[0]
        cluster_rows = arr[indices]
        cluster_angles = utils_math.calculate_abs_angle(cluster_rows, np.expand_dims(clust_centers[label], axis=0))
        cluster_mags, sorted_mags = utils_math.get_norm_with_rank(cluster_rows)
        
        # Get minimum angle 
        min_angle_index = np.argmin(cluster_angles)
        min_angle = cluster_angles[min_angle_index] 

        # Find all angles within a threshold of minimum angle
        selected_angle_indices = np.where(cluster_angles <= min_angle + angle_threshold)[0]
        # Get the highest magnitude for points within this threshold 
        highest_mag_index = np.argmax(cluster_mags[selected_angle_indices])
        # Select that as the index from the cluster 
        selected_indices.append(indices[selected_angle_indices][highest_mag_index])
            
    final_volume = utils_math.cal_vol(arr[selected_indices, :]) 

    return final_volume, sorted(selected_indices)



    

    




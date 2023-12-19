import numpy as np 
from scipy import linalg
import numpy as np 
from sklearn.cluster import KMeans
from collections import defaultdict


def cal_vol(arr, dth_root=True): 
    """
    Get the volume based on the SVD decomposition of input array 
    Args:
        arr (np.ndarray): Input array containing the data 
        dth_root (bool): If true, dth root(volume) is returned where d = number of dimensions used to cal vol 

    Returns:
        volume (float): Product of singular values, i.e, the volumne 
    """

    # Get singular value decomposition (SVD) of the arr 
    s = linalg.svd(arr, compute_uv=False)

    # Product of singular values is the volume 
    volume = np.prod(s) 

    if dth_root: 
        volume = np.power(volume, 1/len(s))

    return volume


def cartesian_to_nsphere(arr):
    """
    Convert cartesian coordinates to n-sphere coordinates 
    Reference: https://en.wikipedia.org/wiki/N-sphere    
    
    Formula: 
        r = sqrt(x1^2 + x^2 + .... xn^2) 
        theta1 = arctan2(sqrt(x2^2 + x3^2 + .. xn^2), x1^2) 
        theta2 = arctan2(sqrt(x3^3 + x4^2 + ... xn^2), x2^2) 
        ..
        thetan-1 = arctan2(sqrt(xn^2), xn-1))

    Args:
        arr (np.ndarray): Input array containing data in cartesian coordinates 

    Returns: 
        r (np.ndarray): Array containing the radial coordinates 
        thetas (np.ndarray): Array containing the angular coordinates
    """

    # Radial coordinate
    r = np.linalg.norm(arr, axis=1) 

    # Angular coordinates 
    thetas = np.zeros((arr.shape[0], arr.shape[1]-1))

    for i in range(1, arr.shape[1]): 
        thetas[:, i-1] = np.arctan2(np.linalg.norm(arr[:, i:], axis=1), arr[:, i-1]) 
    
    
    return r, thetas 


def get_kmeans_clusters(arr, clust_param_dict):
    """
    Perform k-means clustering on the input data 

    Args:
        arr (np.ndarray): Input array 
        clust_param_dict: Dictionary containing k-means hyperparamters. 
            n_clusters (int, required): Number of clusters 
            random_state (int, optional): If provided, clustering is done with a fixed a random seed  
            n_init (str, optional): Initialization strategy. (Recommended='auto')
            init (arr, optional): Specify intial centroids if required 

    Returns:
        labels (np.ndarray): Cluster labels for each point 
        clust_dist (dict): Distribution of points across clusters 
    """

    # Perform k-means clustering 
    kmeans = KMeans(**clust_param_dict).fit(arr)
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    unique, counts = np.unique(labels, return_counts=True)
    clust_dist = dict(zip(unique, counts))

    return labels, clust_dist


def radial_coord_per_cluster(arr, labels): 
    """
    Get radial coordinates for each point in each cluster. Array is expected to be in angular coordinates. 
    Args:
        arr (np.ndarray): Input array (in angular coordinates, i.e,  thetas)
        labels (np.ndarray): Cluster labels for each point  

    Returns:
        r_dict (dict): Dictionary containing radial coordinates for each point in each cluster. {clust_label: radial..}
    """
    r_dict = {} 
    for label in sorted(np.unique(labels)): 
        indices = np.where(labels == label)[0]
        r_dict[label] = [np.linalg.norm(arr[indices, :], axis=1), indices] 
    
    return r_dict


def max_radial_coordinate_per_cluster(r_dict):
    """
    Get the maximum radial coordinate from each cluster and return them as selected indices 
    Args:
        r_dict (dict): Dictionary containing radial coordinates for each point in each cluster. {clust_label: radial..}

    Returns:
        max_per_cluster (list): Value of max radial coordinate from each cluster 
        selected_indices (list): Index of max radial coordinate from each cluster 
    """
    # Get max radial coordinate from each cluster
    selected_indices = []
    max_per_cluster = []
    for label, rvalues in r_dict.items():
        r_coors, indices = rvalues[0], rvalues[1] 
        max_per_cluster.append(np.max(r_coors))
        max_index = np.argmax(r_coors)
        selected_indices.append(indices[max_index])
    
    return max_per_cluster, selected_indices


def get_norm_with_rank(arr): 
    """
    Given an array, return magnitudes and indices sorted based on those magnitudes (in descending order)
    Args:
        arr (np.ndarray): Input data array 

    Returns:
        magnitudes (np.ndarray): Norm of the array 
        sorted_indices (np.ndarray): Sorted indices in descending order based on the norm  
    """
    # Get the norm 
    magnitudes = np.linalg.norm(arr, axis=1) 
    # Sort the magnitudes in descending order 
    sorted_indices = np.argsort(magnitudes)[::-1]

    return magnitudes, sorted_indices


def get_indices_rank(sorted_indices, subset_indices): 
    """
    Given an arr of sorted indices (by magnitude/norm) and a subset array of selected indices, return the corresponding
    rank for the indices in the subset array based on the sorted array. 
    Args:
        sorted_indices (np.ndarray): Complete array of indices, sorted based on their magnitudes 
        subset_indices (np.ndarray): Final indices selected by some algorithm  

    Returns:
        ranks (dict): Dictionary of the type {index: rank....}
    """
    subset_ranks = [np.where(sorted_indices == index)[0][0] for index in subset_indices]
    ranks = dict(sorted(zip(subset_indices, subset_ranks), key=lambda x: x[1]))    
    return ranks 


def get_cluster_for_index(labels, subset_indices): 
    """
    Given cluster labels and selected indices, generate index dict of type {index: clust_label...} and label count dict 
    of type {label: total_count....} 
    Args:
        labels (np.ndarray): Cluster labels
        subset_indices (np.ndarray): Selected indices 
    """
    index_dict = dict(sorted(zip(subset_indices, labels[subset_indices]), key=lambda x: x[1]))    

    label_counts = defaultdict(int)
    for value in index_dict.values():
        label_counts[value] += 1

    return index_dict, label_counts


def flip_signs(arr): 
    """
    Take the input array, check first col of each row. If negative, flip signs for ALL cols in that row. 
    If positive, leave it alone. 

    Args:
        arr (np.ndarray): Input data array 

    Returns:
        arr (np.ndarray): Modified data array with flipped signs 
    """
    negative_mask = arr[:, 0] < 0
    arr[negative_mask, :] *= -1 
    return arr 


def modified_cosine_dist(a, b): 
    """
    Modified cosine distance to move opposite vectors to same quadrant 
    Formula: 1 - (abs(a.b)/(||a||*||b||))  

    The absolute value of the dot product of vectors to bring vectors to same quadrant
    
    Args:
        a (np.ndarray): Vector 1 
        b (np.ndarray): Vector 2 

    Returns:
        distance (float): Modified cosine distance between the two vectors 
    """

    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    distance = 1 - np.abs(dot_product) / (norm_a * norm_b)
    return distance


def modified_cosine_dist_2pair(arr):
    dot_products = np.dot(arr, arr.T) 
    norms = np.linalg.norm(arr, axis=1)
    numerator = np.abs(dot_products) 
    denominator = np.outer(norms, norms) 
    distances = 1 - (numerator/denominator) 
    np.fill_diagonal(distances, 0)

    upper_triangular_indices_dist = np.triu_indices(n=distances.shape[0], m=distances.shape[1], k=1)    
    sorted_distances = np.argsort(distances[upper_triangular_indices_dist])
    print(sorted_distances.shape)
    upper_triangular_indices_norm =  np.triu_indices(n=denominator.shape[0], m=denominator.shape[1], k=1)
    sorted_norms = np.argsort(denominator[upper_triangular_indices_norm])

    selected_dist_indices = np.concatenate((upper_triangular_indices_dist[0][sorted_distances[-13:]], 
                                           upper_triangular_indices_dist[1][sorted_distances[-13:]],
                                            ))
                                           
    selected_norm_indices = np.concatenate((upper_triangular_indices_norm[0][sorted_norms[-13:]], 
                                           upper_triangular_indices_norm[1][sorted_norms[-13:]],
                                           ))
                                           
    return selected_dist_indices, selected_norm_indices
 

def scale_data(arr, technique):
    if technique == 'simple': 
        scaled_arr = arr*10 
    
    return scaled_arr




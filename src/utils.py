import numpy as np 
from scipy import linalg
import numpy as np 
from sklearn.decomposition import PCA
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from constants import DATA_PATH


# Plotting constants 
sns.set_theme(style='whitegrid')


def cal_vol(arr): 
    """
    Get the volume based on the SVD decomposition of input array 
    Args:
        arr (np.ndarray): Input array containing the data 

    Returns:
        volume (float): Product of singular values, i.e, the volumne 
    """

    # Get singular value decomposition (SVD) of the arr 
    U, s, Vh = linalg.svd(arr)
    # Product of singular values is the volume 
    volume = np.prod(s) 

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


def get_norm_with_rank(arr): 
    # Get the norm 
    magnitudes = np.linalg.norm(arr, axis=1) 
    # Sort the magnitudes in descending order 
    sorted_indices = np.argsort(magnitudes)[::-1]

    return magnitudes, sorted_indices


def get_indices_rank(sorted_indices, subset_indices): 
    subset_ranks = [np.where(sorted_indices == index)[0][0] for index in subset_indices]
    ranks = dict(sorted(zip(subset_indices, subset_ranks), key=lambda x: x[1]))    
    return ranks 


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
 

def plot_pca(X, labels, metric, num_components, selected_indices=None, filename=None): 


    # PCA Visualization 
    # Perform the fit on the entire data for accurate results (not on the subset)

    pca = PCA(n_components=num_components).fit(X)
    X_transformed = pca.transform(X) 
    print("PCA Shape", X_transformed.shape)

    num_labels = np.unique(labels).shape[0]
    main_colors = sns.color_palette('husl', num_labels)

    if num_components == 2: 
        fig1, ax1 = plt.subplots()
        df_transformed = pd.DataFrame(X_transformed, columns=['Component1', 'Component2'])
        df_transformed['label'] = labels 
        sns.scatterplot(data=df_transformed, x='Component1', y='Component2', hue='label', palette=main_colors, ax=ax1)

        if selected_indices: 
            df_transformed['selected'] = np.where(np.isin(np.arange(len(df_transformed)), selected_indices),
                                                  'Selected', 'Not Selected')
            
            sns.scatterplot(data=df_transformed[df_transformed['selected'] == 'Selected'], 
                            x='Component1', y='Component2', color='red', ax=ax1, marker='x', s=80, linewidth=2)

        ax1.set_title(f"{metric}")

        if num_labels > 5: 
            ax1.get_legend().remove()

    elif num_components == 3: 
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Create a color map
        cmap = ListedColormap(main_colors)

        # Map each label value to a color
        colors_array = cmap(labels)
        ax.scatter(X_transformed[:, 0], X_transformed[:, 1], X_transformed[:, 2], c=colors_array)       
        ax.set_xlabel('Component1')
        ax.set_ylabel('Component2')
        ax.set_zlabel('Component3')

        # Create a legend with labels and corresponding colors
        legend_labels = {label: color for label, color in zip(labels, colors_array)}
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=label, 
                                    markerfacecolor=color, markersize=10) for label, color in legend_labels.items()]
        ax.legend(handles=legend_elements, title="Legend", loc='upper right')

    if filename is not None: 
        fig1.savefig(DATA_PATH / filename)
        plt.close(fig1)


def scale_data(arr, technique):
    if technique == 'simple': 
        scaled_arr = arr*10 
    
    return scaled_arr


def plot_tsne(X, labels): 
    fig1, ax1 = plt.subplots()
    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(X)
    print("KL Divergence Val: ", tsne.kl_divergence_)
    df_tsne = pd.DataFrame(X_tsne, columns=['Component1', 'Component2'])
    df_tsne['label'] = labels 
    sns.scatterplot(data=df_tsne, x='Component1', y='Component2', hue='label', palette=colors, ax=ax1)
    ax1.set_title(" T-SNE Visualization") 





from sklearn.decomposition import PCA
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
import numpy as np 
from matplotlib.ticker import MaxNLocator


# Plotting constants 
sns.set_theme(style='whitegrid')


def plot_pca(X, labels, metric, num_components, selected_indices=None, filename=None): 
    """
    Perform principal component analysis (PCA) on the given data 
    Args:
        X (np.ndarray): Input data array 
        labels (np.ndarray): Clustering labels 
        metric (str): A string label which would be used as the plot title. Can be used to provide more info 
        num_components (int): Number of principal components (Valid values = 2 or 3) 
        selected_indices (np.ndarray, optional): If provided, plot the selected indices separately. Defaults to None.
        filename (str/path, optional): Save the plot if filename is provided. Defaults to None.
    """

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
        fig1.savefig(filename)
        plt.close(fig1)


def plot_tsne(X, labels): 
    """
    Perform t-Distributed Stochastic Neighbor Embedding (t-SNE) visualization 
    Args:
        X (np.ndarray): Input data array 
        labels (np.ndarray): Clustering labels 
    """

    num_labels = np.unique(labels).shape[0]
    main_colors = sns.color_palette('husl', num_labels)
    fig1, ax1 = plt.subplots()
    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(X)
    print("KL Divergence Val: ", tsne.kl_divergence_)
    df_tsne = pd.DataFrame(X_tsne, columns=['Component1', 'Component2'])
    df_tsne['label'] = labels 
    sns.scatterplot(data=df_tsne, x='Component1', y='Component2', hue='label', palette=main_colors, ax=ax1)
    ax1.set_title(" T-SNE Visualization") 


def plot_clust_index_dist(cart_counts, ang_counts, filename, algo_info):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5)) 

    # Subplot for cartesian clustering 
    labels1, counts1 = zip(*cart_counts.items())
    axs[0].bar(labels1, counts1)
    axs[0].set_xlabel('Clusters')
    axs[0].set_ylabel('Total Indices within cluster')
    axs[0].set_title(f'{algo_info}: Cartesian Clust')
    axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))

    # Subplot for angular clustering
    labels2, counts2 = zip(*ang_counts.items())
    axs[1].bar(labels2, counts2)
    axs[1].set_xlabel('Clusters')
    axs[1].set_ylabel('Total Indices within cluster')
    axs[1].set_title(f'{algo_info}: Ang Clust')
    axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()  # Adjust layout for better spacing

    fig.savefig(filename, dpi=300)


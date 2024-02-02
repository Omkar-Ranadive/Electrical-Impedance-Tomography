from constants import DATA_PATH, EXP_PATH
import time 
import utils_math
import scipy.io as sio
import logging 
import os 
from datetime import datetime
import argparse 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 

sns.set_theme(style='whitegrid')

parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, required=True)

args = parser.parse_args()

# files = {'clustering_angle.csv': 'Clust_angle', 'clustering_mag.csv': 'Clust_mag', 'clustering_rad.csv': 'Ang_Rad',
#          'clustering_ang5_mag.csv': 'Clust_ang5_mag', 
#          'clustering_ang10_mag.csv': 'Clust_ang10_mag', 'clustering_ang15_mag.csv': 'Clust_ang15_mag',
#          'clustering_ang30_mag.csv': 'Clust_ang30_mag', 'clustering_ang40_mag.csv': 'Clust_ang40_mag.csv',
#          'EnsembleGenetic.csv': 'EnsembleGenetic'}

# files = {'clustering_angle.csv': 'Clust_angle', 'clustering_mag.csv': 'Clust_mag', 
#           'clustering_ang40_mag.csv': 'Clust_ang40_mag.csv',  'EnsembleGenetic.csv': 'EnsembleGenetic'}

# files = {'clustering_angle.csv': 'Clust_angle', 'clustering_ang40_20mag.csv': 'Clust_ang40_20mag', 
#           'clustering_ang40_mag.csv': 'Clust_ang40_mag',  'EnsembleGenetic.csv': 'EnsembleGenetic'}

files = {'clustering_ang40_20mag.csv': 'Clust_ang40_20mag', 
          'EnsembleGenetic.csv': 'EnsembleGenetic', 
          'clustering_ang40_20mag_v4.csv': 'Clust_ang40_20mag_v4'}


EXP_DIR = EXP_PATH / 'OptimizationMatrices'
SAVE_DIR = EXP_DIR / args.exp_name
os.makedirs(SAVE_DIR, exist_ok=True)

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))
fig3, ax3 = plt.subplots(1, 1, figsize=(10, 8)) 
fig4, ax4 = plt.subplots(1, 1, figsize=(10, 8))
colors = sns.color_palette('husl', len(files))
marker_styles = ['o', 's', '^', 'D', 'v', '>', '<', 'p', '*']
i = 0 

for file, name in files.items(): 
    df = pd.read_csv(EXP_DIR / file) 
    volumes = df['V^(1/D)'] 
    marker = marker_styles[i % len(marker_styles)]

    ax.plot(range(0, len(volumes)), volumes, label=name, alpha=0.7, c=colors[i], marker=marker)
    ax2.scatter(range(0, len(volumes)), volumes, label=name, alpha=0.7, color=colors[i], marker=marker)

    # Plot values sorted by measurements (D) 
    df_sorted = df.sort_values(by='D: Measurements')
    volumes_sorted = df_sorted['V^(1/D)']
    ax3.scatter(df_sorted['D: Measurements'], volumes_sorted, label=name, alpha=0.7, color=colors[i], marker=marker)

    # Plot time taken (sorted by measurements (D))
    if 'Elasped Time' in df_sorted.columns: 
        ax4.plot(df_sorted['D: Measurements'], df_sorted['Elasped Time'], label=name, alpha=0.7, 
                 c=colors[i], marker=marker)

    i += 1

ax.set_title('Volume Compare')
ax.set_xlabel('Matrices')
ax.set_ylabel('Sensitivity Volume (V^(1/D))')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
fig.tight_layout()
fig.savefig(SAVE_DIR / 'volume_compare_all.png', dpi=300)

ax2.set_title('Volume Compare')
ax2.set_xlabel('Matrices')
ax2.set_ylabel('Sensitivity Volume (V^(1/D))')
ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))
fig2.tight_layout()
fig2.savefig(SAVE_DIR / 'volume_compare_scatter_all.png', dpi=300)

ax3.set_title('Volume Compare')
ax3.set_xlabel('Measurements (D)')
ax3.set_ylabel('Sensitivity Volume (V^(1/D))')
ax3.legend(loc='upper left', bbox_to_anchor=(1, 1))
fig3.tight_layout()
fig3.savefig(SAVE_DIR / 'volume_compare_sorted_m_all.png', dpi=300)

ax4.set_title('Time Comparison')
ax4.set_xlabel('Measurements (D)')
ax4.set_ylabel('Time taken (in seconds)')
ax4.legend(loc='upper left', bbox_to_anchor=(1, 1))
fig4.tight_layout()
fig4.savefig(SAVE_DIR / 'volume_compare_sorted_m_time_all.png', dpi=300)
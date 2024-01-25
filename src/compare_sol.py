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


files = {'clustering_angle.csv': 'Clust_angle', 'clustering_mag.csv': 'Clust_mag', 'clustering_rad.csv': 'Ang_Rad',
         'clustering_ang5_mag.csv': 'Clust_ang5_mag', 
         'clustering_ang10_mag.csv': 'Clust_ang10_mag', 'clustering_ang15_mag.csv': 'Clust_ang15_mag',
         'EnsembleGenetic.csv': 'EnsembleGenetic'}



EXP_DIR = EXP_PATH / 'OptimizationMatrices' 
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))
colors = sns.color_palette('husl', len(files))
i = 0 
for file, name in files.items(): 
    df = pd.read_csv(EXP_DIR / file) 
    volumes = df['V^(1/D)'] 
    ax.plot(range(0, len(volumes)), volumes, label=name, alpha=0.8, c=colors[i])
    ax2.scatter(range(0, len(volumes)), volumes, label=name, alpha=0.8, color=colors[i])
    i += 1

ax.set_title('Volume Compare')
ax.set_xlabel('Matrices')
ax.set_ylabel('Sensitivity Volume (V^(1/D))')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
fig.tight_layout()
fig.savefig(EXP_DIR / 'volume_compare_all.png', dpi=300)

ax2.set_title('Volume Compare')
ax2.set_xlabel('Matrices')
ax2.set_ylabel('Sensitivity Volume (V^(1/D))')
ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))
fig2.tight_layout()
fig2.savefig(EXP_DIR / 'volume_compare_scatter_all.png', dpi=300)



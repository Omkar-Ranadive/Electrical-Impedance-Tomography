import utils 
from constants import DATA_PATH, EXP_PATH
import scipy.io as sio
import numpy as np 
import algos 
from genetic_algo import Genetic
import logging 
import argparse 
import os 
from datetime import datetime

def print_results(vol, indices, rank_indices): 
    logger.info(f"Max volume: {vol}")
    logger.info(f"Selected Indices: {indices}")
    logger.info("Rank of selected indices: ")
    logger.info(rank_indices)
    logger.info("*"*20) 


parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", required=True, type=str)
args = parser.parse_args()

EXP_DIR = EXP_PATH / args.exp_name 
os.makedirs(EXP_DIR, exist_ok=True)

# Set up logger 
logging.basicConfig(level=logging.INFO, filename=str(EXP_DIR / 'info.log'), format='%(message)s')
logger=logging.getLogger() 
time_stamp = datetime.now()
logger.info(f"Running main.py: {time_stamp.strftime('%H:%M:%S')}") 

# Load the data 
data = sio.loadmat(DATA_PATH / '9ContactsMatrix.mat')
logger.info(data.keys()) 
arr = data['JGQ']
logger.info(f"Shape of input array: {arr.shape}")
logger.info("*"*20) 

# Scale/normalize the data 
arr = utils.scale_data(arr, technique='simple')

# Hyperparameters 
seed = 0 
num_entries = 26

magnitudes, sorted_indices = utils.get_norm_with_rank(arr) 
logger.info("Indices ranked by magnitude")
logger.info(sorted_indices)
logger.info("*"*20)

logger.info("Highest magnitudes approach: ")
logger.info(f"Max volume: {utils.cal_vol(arr[sorted_indices[:num_entries]])}")
logger.info(f"Selected Indices: {sorted_indices[:num_entries]}")
logger.info("*"*20) 

logger.info("Greedy approach: ") 
greedy_vol, greedy_indices = algos.greedy_approach(arr, num_entries=num_entries) 
greedy_mag_ranks = utils.get_indices_rank(sorted_indices, greedy_indices)
print_results(greedy_vol, greedy_indices, greedy_mag_ranks)

logger.info("Max radial coordinate approach: ")
param_dict = {'n_clusters': num_entries, 'n_init': 'auto', 'random_state': seed}
max_rad_vol, max_rad_indices, clust_labels = algos.max_radial_coors_approach(arr, param_dict) 
max_rad_ranks = utils.get_indices_rank(sorted_indices, max_rad_indices)
print_results(max_rad_vol, max_rad_indices, max_rad_ranks)

logger.info("Max radial coordinate approach with flipped signs: ")
arr_flipped = utils.flip_signs(arr)
max_rad_vol_flip, max_rad_indices_flip, clust_labels_flip = algos.max_radial_coors_approach(arr_flipped, param_dict) 
max_rad_flip_ranks = utils.get_indices_rank(sorted_indices, max_rad_indices_flip)
print_results(max_rad_vol_flip, max_rad_indices_flip, max_rad_flip_ranks)

logger.info("Max radial coordinates with each axis as init centroid")
initial_centers = np.eye(N=num_entries, M=num_entries-1)
param_dict = {'n_clusters': num_entries, 'n_init': 'auto', 'random_state': seed, 'init': initial_centers}
max_rad_vol_axis, max_rad_indices_axis, clust_labels_axis = algos.max_radial_coors_approach(arr, param_dict) 
max_rad_axis_ranks = utils.get_indices_rank(sorted_indices, max_rad_indices_axis)
print_results(max_rad_vol_axis, max_rad_indices_axis, max_rad_axis_ranks)

logger.info("Random volume approach (avg over 1000 iterations)")
ran_vol = algos.random_approach(arr, num_entries=num_entries) 
logger.info(f"Avg random volume: {ran_vol}")
logger.info("*"*20) 

# print("Genetic algorithm approach")
# genObj = Genetic(arr=arr, num_generations=500, population_size=10000, 
#                  num_parents=4000, mutation_rate=0.1, print_every=100)
# best_results = genObj.run_genetic_simulation() 
# print(f"Max Volume: {best_results['best_score']}")
# print(f"Selected Indices: {best_results['best_indices']}")
# print(f"Best Generation: {best_results['best_generation']}")
# print("*"*20) 
logger.info("Modified Cosine: ") 
max_cos_vol, max_cos_indices, clust_labels = algos.cosine_clustering(arr)
max_cos_ranks = utils.get_indices_rank(sorted_indices, max_cos_indices)
print_results(max_cos_vol, max_cos_indices, max_cos_ranks)

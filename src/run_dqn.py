import algo_dqn 
import scipy.io as sio
from constants import DATA_PATH, EXP_PATH
import utils_math
import torch.optim
import logging 
import argparse
import matplotlib.pyplot as plt 
import seaborn as sns 
from utils import save_friendly
from matplotlib.ticker import MaxNLocator
import time 


sns.set_theme(style='whitegrid')

parser = argparse.ArgumentParser()

# Hyper-parameters for training 
parser.add_argument("--episodes", type=int, required=True, help="Number of episodes to run") 
parser.add_argument("--lr", default=1e-3, type=float, help="Learning Rate") 
parser.add_argument("--batch_size", type=int, default=512, help="Training batch size") 
parser.add_argument("--tfreq", default=100, type=int, help="Target model update frequency") 
parser.add_argument("--gamma", type=float, default=0.98, help="Discount factor")
parser.add_argument("--epsilon", type=float, default=1.0, help="Exploration parameter")
parser.add_argument("--dr", type=float, default=0.95, help="Decay rate of epsilon")

args = parser.parse_args()

C, M, D = 8, 28, 20

data = sio.loadmat(DATA_PATH / 'OptimizationMatrices'  / f'{C}contacts_{M}polys.mat')
EXP_DIR = EXP_PATH /  'OptimizationMatrices' / f'{C}contacts_{M}polys_D{D}' 
arr = data['JGQ']
arr = utils_math.scale_data(arr, technique='simple')
print("Shape of data array: ", arr.shape)

num_entries = D 
filename = (f'dqn_e{args.episodes}_b{args.batch_size}_l{save_friendly(args.lr)}'
            f'_g{save_friendly(args.gamma)}_t{save_friendly(args.tfreq)}'
            f'_e{save_friendly(args.epsilon)}_dr{save_friendly(args.dr)}')

logging.basicConfig(level=logging.INFO, filename=(EXP_DIR / f'{filename}').with_suffix('.log'), 
                    format='%(message)s')
logger = logging.getLogger() 
logger.info("*"*20)
logger.info(f"Array: {C}contacts_{M}polys_{D}, Shape: {arr.shape}") 
logger.info(f'Measurements (num entries): {num_entries}')
logger.info("*"*5)
logger.info(f"Hyperparameters")
logger.info(f"Episodes: {args.episodes}, Learning Rate: {args.lr}, Batch Size: {args.batch_size}")
logger.info(f"Target update Freq: {args.tfreq}, Discount Factor: {args.gamma}")
logger.info(f'Epsilon: {args.epsilon}, Decay Rate: {args.dr}')


env = algo_dqn.VolumeMaximizationEnv(arr, num_entries=num_entries)
model = algo_dqn.DQN(env.state_size, env.action_size)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
start_time = time.time()
best_dict, model = algo_dqn.train_dqn_agent(env, model, optimizer, num_episodes=args.episodes, gamma=args.gamma, 
                                     target_update_freq=args.tfreq, batch_size=args.batch_size, epsilon=args.epsilon, 
                                     decay_rate=args.dr)

logger.info(f"Best Volume {best_dict['best_volume']}")
logger.info(f"Best Episode: {best_dict['best_episode']}")
logger.info(f"Best Indices: {best_dict['best_indices']}")
elapsed_time = (time.time() - start_time)/60  # Time in minutes 
logger.info(f"Training time: {elapsed_time} minutes")

fig, ax = plt.subplots(1, 2, figsize=(12, 8)) 

ax[0].plot(range(0, args.episodes), best_dict['rewards'], label="Rewards")
ax[0].set_xlabel('Episodes')
ax[0].set_ylabel('Average Reward')

ax[1].plot(range(0, args.episodes), best_dict['volumes'], label='Volumes') 
ax[1].set_xlabel('Episodes')
ax[1].set_ylabel('Average Volume')

ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
ax[0].legend()
ax[1].legend()

fig.tight_layout()
fig.savefig((EXP_DIR / filename).with_suffix('.png'), dpi=300)
import numpy as np
from numpy.linalg import linalg
import torch
import torch.nn as nn
import torch.optim as optim
from utils_math import cal_vol


class VolumeMaximizationEnv:
    def __init__(self, arr, num_entries):
        self.arr = arr
        self.num_entries = num_entries
        self.n, self.m = arr.shape
        self.available_rows = set(range(self.n))
        self.selected_rows = []
        self.current_volume = 1
        self.state_size = self.n + self.n*self.m + 1    # Selected Indices + Flattened Array + Current Volume
        self.action_size = self.n   # Action size = number of rows 

    def reset(self):
        self.available_rows = set(range(self.n))
        self.selected_rows = []
        self.current_volume = 1
        return self._get_state()

    def step(self, action):
        if action not in self.available_rows:  # Handle invalid action
            return self._get_state(), -1, False

        next_index = action
        self.selected_rows.append(next_index)
        self.available_rows.remove(next_index)
        subset_matrix = self.arr[self.selected_rows, :]
        next_volume = cal_vol(subset_matrix)
        reward = next_volume - self.current_volume
        self.current_volume = next_volume
        done = len(self.selected_rows) == self.num_entries
        return self._get_state(), reward, done

    def _get_state(self):
        selected_indices = np.zeros(self.n)
        if self.selected_rows:
            selected_indices[self.selected_rows] = 1
        return np.concatenate(([self.current_volume], selected_indices, self.arr.flatten()))
    
    def get_results(self): 
        return self.current_volume, self.selected_rows
    
    def sample_action(self): 
        return np.random.choice(np.array(list(self.available_rows)), size=1)[0]
    

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)  
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))  
        return self.fc3(x)


class ReplayBuffer(): 
    def __init__(self): 
        self.rewards = []
        self.actions = []
        self.states = []
        self.next_states = []
        self.dones = []
    
    def append(self, reward, action, state, next_state, done):
        self.rewards.append(reward)
        self.actions.append(action)
        self.states.append(state)
        self.next_states.append(next_state)
        self.dones.append(done)
    
    def __len__(self):
        return len(self.rewards)
    
    def get_minibatch(self, batch_size):
        minibatch_indices = np.random.choice(len(self.rewards), size=batch_size)

        rewards_batch = np.array(self.rewards)[minibatch_indices]
        actions_batch = np.array(self.actions)[minibatch_indices]
        states_batch = np.array(self.states)[minibatch_indices]
        next_states_batch = np.array(self.next_states)[minibatch_indices]
        dones_batch = np.array(self.dones)[minibatch_indices]

        return rewards_batch, actions_batch, states_batch, next_states_batch, dones_batch


def train_dqn_agent(env, model, optimizer, num_episodes, gamma, target_update_freq, batch_size, 
                    epsilon, decay_rate, max_it=10000):
    target_model = DQN(env.state_size, env.action_size)
    target_model.load_state_dict(model.state_dict())

    if torch.cuda.is_available():
        device = "cuda"
        model.to(device)
        target_model.to(device)
    else:
        device = "cpu"

    replay_buffer = ReplayBuffer()
    best_dict = {'best_volume': -np.inf, 'best_indices': None, 'best_episode': -1, 'rewards': [], 'volumes': []}
    volumes_per_episode = [] 
    rewards_per_episode = []
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        rewards_within_episode = []
        counter = 0 

        while not done or counter >= max_it:
            if np.random.uniform() < epsilon: 
                action = env.sample_action() 
            else:
                action_values = model(torch.tensor(state, dtype=torch.float32).view(1, -1).to(device))
                action = torch.argmax(action_values, dim=1).cpu().item()
            next_state, reward, done = env.step(action)
            # next_state = torch.tensor(next_state, dtype=torch.float32).view(1, -1).to(device)
            rewards_within_episode.append(reward)
            # print(counter, state.shape,  next_state.shape, action, reward, done)
            # print(counter, state.dtype,  next_state.dtype, action, reward, done)

            replay_buffer.append(reward, action, state, next_state, done)
            counter += 1 

            # Decay the epsilon 
            epsilon = max(0.05, epsilon*decay_rate)

            if len(replay_buffer) >= 100:
                rewards, actions, states, next_states, dones = replay_buffer.get_minibatch(batch_size=batch_size)

                states = torch.tensor(states, dtype=torch.float32).to(device)
                actions = torch.tensor(actions, dtype=torch.long).to(device)
                rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
                next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
                dones = torch.tensor(dones, dtype=torch.float32).to(device)

                q_values = model(states)
                next_q_values = target_model(next_states)
                max_next_actions = next_q_values.argmax(dim=1, keepdim=True)
                next_q_values_target = target_model(next_states)[range(batch_size), max_next_actions.squeeze()]
                expected_q_values = rewards + gamma * next_q_values_target * (1 - dones)
                # print(q_values.size(), actions.size(), expected_q_values.size())
                loss = nn.functional.smooth_l1_loss(q_values.gather(1, actions.unsqueeze(1)), 
                                                    expected_q_values.unsqueeze(1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if episode % target_update_freq == 0:
                    target_model.load_state_dict(model.state_dict())
        
        volume, selected_rows = env.get_results() 
        volumes_per_episode.append(volume)
        if best_dict['best_volume'] < volume: 
            best_dict['best_volume'] = volume
            best_dict['best_indices'] = selected_rows
            best_dict['best_episode'] = episode
              
        rewards_per_episode.append(np.mean(rewards_within_episode))

        if episode % 10 == 0:
            avg_reward = np.mean(rewards_per_episode)
            avg_vol = np.mean(volumes_per_episode)
            print(f"Episode: {episode}, Average Reward: {avg_reward}, Average Volume: {avg_vol}")
            print(f"Best Volume found at episode {episode}: {best_dict['best_volume']}")
    
    volume, selected_rows = env.get_results() 
    print(f"Volume at last episode: {volume}")
    print(best_dict)

    best_dict['volumes'] = volumes_per_episode
    best_dict['rewards'] = rewards_per_episode

    return best_dict, model



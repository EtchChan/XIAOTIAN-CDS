# radar_tracking.py

import pandas as pd
import numpy as np
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

from sympy.physics.units import velocity

from src.GNN.GAT_classifier import predict

# detect if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================
# Data Preprocessing
# ==========================

def load_data(filepath):
    # Load the data
    data = pd.read_csv(filepath)

    # Convert column names to English for ease of use
    data.columns = ['Time', 'SlantRange', 'Azimuth', 'Elevation', 'RadialVelocity', 'Circle']

    # calculate the x, y, z coordinates and add them to the dataframe
    data['X'] = data['SlantRange'] * np.cos(np.radians(data['Azimuth'])) * np.cos(np.radians(data['Elevation']))
    data['Y'] = data['SlantRange'] * np.sin(np.radians(data['Azimuth'])) * np.cos(np.radians(data['Elevation']))
    data['Z'] = data['SlantRange'] * np.sin(np.radians(data['Elevation']))

    return data

# ==========================
# Environment Definition
# ==========================

class RadarTrackingEnv(gym.Env):
    def __init__(self, data):
        super(RadarTrackingEnv, self).__init__()

        self.data = data
        self.unique_circles = self.data['Circle'].unique()
        self.current_circle_index = 0
        self.max_circle_index = len(self.unique_circles) - 1

        # Define maximum number of tracks and detections
        self.max_tracks = 10
        self.max_detections = 300

        # Define action and observation spaces
        # Action space: For detections in the current circle, pick one and assign to a track or start a new track
        self.action_space = spaces.MultiDiscrete([self.max_tracks + 1] * self.max_detections)

        # Observation space: Positions, velocities and time stamps of detections and tracks
        self.observation_space = spaces.Dict({
            'detections': spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_detections, 5), dtype=np.float32),
            'tracks': spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_tracks, 5), dtype=np.float32),
        })

        # Initialize state variables
        self.tracks = {}  # Track ID --> Track State
        self.track_id_counter = 0

    def reset(self):
        self.current_circle_index = 0
        self.tracks = {}
        self.track_id_counter = 0

        return self._get_observation()

    def _get_observation(self):
        current_circle = self.unique_circles[self.current_circle_index]
        detections = self.data[self.data['Circle'] == current_circle]
        detection_states = detections[['X', 'Y', 'Z', 'RadialVelocity', 'Time']].values

        # Pad or truncate detections to max_detections
        if len(detection_states) < self.max_detections:
            padding = np.zeros((self.max_detections - len(detection_states), 5))
            detection_states = np.vstack((detection_states, padding))
        else:
            detection_states = detection_states[:self.max_detections]

        # Get track states
        track_states = np.zeros((self.max_tracks, 5))
        for i, (track_id, track_state) in enumerate(self.tracks.items()):
            if i >= self.max_tracks:
                break
            track_states[i] = track_state

        observation = {
            'detections': detection_states,
            'tracks': track_states
        }
        return observation

    def step(self, action):
        # Get current detections
        current_circle = self.unique_circles[self.current_circle_index]
        detections = self.data[self.data['Circle'] == current_circle]
        detection_states = detections[['X', 'Y', 'Z', 'RadialVelocity', 'Time']].values

        # Apply action
        reward = self._compute_reward(action, detection_states)

        # Update tracks based on action
        self._update_tracks(action, detection_states)

        # Move to next time step
        self.current_circle_index += 1
        done = self.current_circle_index >= self.max_circle_index

        if not done:
            obs = self._get_observation()
        else:
            obs = self._get_observation()  # Return the last observation

        return obs, reward, done, {}

    def _compute_reward(self, action, detection_states):
        reward = 0.0

        # For each detection
        for i, track_assignment in enumerate(action):
            if i >= len(detection_states):
                break  # No more detections

            detection = detection_states[i]
            assigned_track = track_assignment.item()

            # If assigned to an existing track
            if assigned_track < self.max_tracks:
                if assigned_track > len(self.tracks):
                    # create a new track
                    track_id = self.track_id_counter
                    self.track_id_counter += 1
                    self.tracks[track_id] = detection
                    reward -= 5.0  # Penalize creating new tracks to avoid arbitrary track creation
                else:
                    track_ids = list(self.tracks.keys())
                    if assigned_track >= len(track_ids):
                        continue  # Invalid track assignment

                    track_id = track_ids[assigned_track]
                    previous_state = self.tracks[track_id]

                    # gain reward for maintaining existing tracks
                    reward += 5.0

                    # Compute distance between detection and predicted track state
                    dt = detection[4] - previous_state[4]
                    tolerance_range = previous_state[3] * dt * 10.0  # Tolerance range based on velocity
                    position_distance = np.linalg.norm(detection[:3] - previous_state[:3])
                    velocity_difference = np.abs(detection[3] - previous_state[3])

                    # Compute reward
                    # Reward closer detections
                    if position_distance > tolerance_range:
                        if position_distance < 200: # Distance threshold
                            reward -= position_distance * 0.1  # Penalize distance
                        else:
                            reward -= (20 + position_distance * 0.15) # Penalize unphysical distances
                    else:
                        reward += 1.5 # Reward similar positions
                    # Reward similar velocities
                    if velocity_difference > 10:  # Velocity threshold
                        reward -= (15.0 + velocity_difference * 1.5)# Penalize large velocity differences
                    elif velocity_difference > 5:
                        reward -= velocity_difference * 0.5  # Penalize large velocity changes
                    else:
                        reward += 1.0  # Reward similar velocities
            else:
                # consider the point as noise
                reward += 15.0  # encourage ignoring noise

        # Small positive reward for each maintained track
        reward += len(self.tracks) * 0.8

        return reward

    def _update_tracks(self, action, detection_states):
        # Temporary dictionary to hold updated tracks
        updated_tracks = {}

        # For each detection
        for i, track_assignment in enumerate(action):
            if i >= len(detection_states):
                break  # No more detections

            detection = detection_states[i]
            assigned_track = track_assignment

            # If assigned to an existing track
            if assigned_track < self.max_tracks:
                if assigned_track > len(self.tracks):
                    # create a new track
                    track_id = self.track_id_counter
                    self.track_id_counter += 1
                    self.tracks[track_id] = detection
                else:
                    track_ids = list(self.tracks.keys())
                    if assigned_track >= len(track_ids):
                        continue  # Invalid track assignment

                    track_id = track_ids[assigned_track]
                    # Update track state with the new detection
                    updated_tracks[track_id] = detection
            else:
                # consider the point as noise, just pass it
                pass

        # Update tracks with the new states
        self.tracks = updated_tracks

# ==========================
# Agent Definition with Multi-Head Attention
# ==========================

class DQNAgent(nn.Module):
    def __init__(self, detection_size, track_size, action_size, n_heads=4):
        super(DQNAgent, self).__init__()

        self.detection_input_dim = detection_size * 5
        self.track_input_dim = track_size * 5
        self.embedding_dim = 64

        # Embedding layers
        self.detection_embedding = nn.Linear(self.detection_input_dim, self.embedding_dim)
        self.track_embedding = nn.Linear(self.track_input_dim, self.embedding_dim)

        # Multi-Head Attention
        self.multihead_attention = nn.MultiheadAttention(embed_dim=self.embedding_dim, num_heads=n_heads, batch_first=True)

        hidden_dim = 64
        # Fully connected layers after attention
        self.fc1 = nn.Linear(self.embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, detection_size *  self.action_per_detection())

    def action_per_detection(self):
        # Since each detection can be assigned to one of the tracks or a new track
        return env.max_tracks + 1

    def forward(self, state_dict):
        detections = state_dict['detections']
        tracks = state_dict['tracks']

        # Flatten detections and tracks
        batch_size = detections.size(0)

        detections_flat = detections.view(batch_size, -1)
        tracks_flat = tracks.view(batch_size, -1)

        # Create embeddings
        detection_emb = self.detection_embedding(detections_flat)
        track_emb = self.track_embedding(tracks_flat)

        # Reshape for attention (batch_size, seq_length, embedding_dim)
        detection_emb = detection_emb.unsqueeze(1)
        track_emb = track_emb.unsqueeze(1)

        # Combine embeddings
        embeddings = torch.cat((detection_emb, track_emb), dim=1)

        # Apply multi-head attention
        attn_output, _ = self.multihead_attention(embeddings, embeddings, embeddings)

        # Take the first output corresponding to detections
        attn_output = attn_output[:, 0, :]  # Shape: (batch_size, embedding_dim)

        # Pass through fully connected layers
        x = torch.relu(self.fc1(attn_output))
        x = torch.relu(self.fc2(x))
        x = self.out(x)

        # Reshape output to (batch_size, num_detections, num_actions_per_detection)
        x = x.view(batch_size, env.max_detections, -1)

        return x

# ==========================
# Training Function
# ==========================

def train_agent(env, agent, episodes=50, tolerance=30):
    optimizer = optim.Adam(agent.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    replay_buffer = deque(maxlen=5000)
    batch_size = 32
    gamma = 0.99
    epsilon = 1.0  # Exploration rate
    epsilon_decay = 0.995
    epsilon_min = 0.01

    # Early stopping parameters
    best_reward = -np.inf
    patience = 0

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Epsilon-greedy action selection
            if np.random.rand() <= epsilon:
                # Random action, confine num of points picked to the max number of tracks
                # initialize the action with all max_tracks
                action = np.full(env.max_detections, env.max_tracks)
                # then randomly assign some points to the tracks
                for i in range(env.max_detections):
                    if np.random.rand() < env.max_tracks * 3 / env.max_detections:
                        action[i] = np.random.randint(env.max_tracks)
            else:
                # Use the DQN to get action
                # Prepare state tensors
                detections = torch.FloatTensor(state['detections']).unsqueeze(0)  # Shape: (1, max_detections, 5)
                tracks = torch.FloatTensor(state['tracks']).unsqueeze(0)  # Shape: (1, max_tracks, 5)
                # if gpu is available, move the tensors to gpu
                detections = detections.to(device)
                tracks = tracks.to(device)
                state_tensor = {'detections': detections, 'tracks': tracks}

                q_values = agent(state_tensor)  # Shape: (1, max_detections, num_actions_per_detection)
                q_values = q_values.squeeze(0)  # Shape: (max_detections, num_actions_per_detection)

                # Select actions with highest Q-value for each detection
                action = q_values.argmax(dim=1).cpu().numpy()


            # Take action
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            # Store experience in replay buffer
            replay_buffer.append((state, action, reward, next_state, done))

            # Learn from replay buffer
            if len(replay_buffer) > batch_size:
                batch = random.sample(replay_buffer, batch_size)
                states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch = zip(*batch)

                # Prepare tensors
                # state_detections = torch.FloatTensor([s['detections'] for s in states_batch])
                # state_tracks = torch.FloatTensor([s['tracks'] for s in states_batch])
                # next_state_detections = torch.FloatTensor([ns['detections'] if ns is not None else np.zeros_like(s['detections']) for s, ns in zip(states_batch, next_states_batch)])
                # next_state_tracks = torch.FloatTensor([ns['tracks'] if ns is not None else np.zeros_like(s['tracks']) for s, ns in zip(states_batch, next_states_batch)])

                # before convert to tensor, convert the list to np array to make the conversion more efficient
                state_detections = torch.FloatTensor(np.array([s['detections'] for s in states_batch]))
                state_tracks = torch.FloatTensor(np.array([s['tracks'] for s in states_batch]))
                next_state_detections = np.array([ns['detections'] if ns is not None else np.zeros_like(s['detections']) for s, ns in zip(states_batch, next_states_batch)])
                next_state_tracks = np.array([ns['tracks'] if ns is not None else np.zeros_like(s['tracks']) for s, ns in zip(states_batch, next_states_batch)])
                next_state_detections = torch.FloatTensor(next_state_detections)
                next_state_tracks = torch.FloatTensor(next_state_tracks)

                actions_tensor = torch.LongTensor(actions_batch)  # Shape: (batch_size, max_detections)
                rewards_tensor = torch.FloatTensor(rewards_batch)
                dones_tensor = torch.FloatTensor(dones_batch)

                # if gpu is available, move the tensors to gpu
                state_detections = state_detections.to(device)
                state_tracks = state_tracks.to(device)
                next_state_detections = next_state_detections.to(device)
                next_state_tracks = next_state_tracks.to(device)
                actions_tensor = actions_tensor.to(device)
                rewards_tensor = rewards_tensor.to(device)
                dones_tensor = dones_tensor.to(device)


                # Compute Q-values
                q_values = agent({'detections': state_detections, 'tracks': state_tracks})  # Shape: (batch_size, max_detections, num_actions_per_detection)
                q_values = q_values.gather(2, actions_tensor.unsqueeze(2)).squeeze(2)  # Shape: (batch_size, max_detections)

                # Compute target Q-values
                with torch.no_grad():
                    next_q_values = agent({'detections': next_state_detections, 'tracks': next_state_tracks})
                    max_next_q_values, _ = next_q_values.max(dim=2)  # Shape: (batch_size, max_detections)
                    targets = rewards_tensor.unsqueeze(1) + gamma * max_next_q_values * (1 - dones_tensor.unsqueeze(1))

                # Compute loss
                loss = criterion(q_values, targets)

                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            state = next_state

        # Decay epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.4f}")

        if total_reward > best_reward:
            best_reward = total_reward
            patience = tolerance
        else:
            patience -= 1
            if patience == 0:
                print(f"Early stopping at episode {episode+1}/{episodes}")
                break



# ==========================
# Evaluation Function
# ==========================

def evaluate_agent(env, agent):
    state = env.reset()
    done = False
    tracks_output = {}

    while not done:
        # Use the DQN to get action
        # Prepare state tensors
        detections = torch.FloatTensor(state['detections']).unsqueeze(0)  # Shape: (1, max_detections, 5)
        tracks = torch.FloatTensor(state['tracks']).unsqueeze(0)  # Shape: (1, max_tracks, 5)

        # if gpu is available, move the tensors to gpu
        detections = detections.to(device)
        tracks = tracks.to(device)
        state_tensor = {'detections': detections, 'tracks': tracks}

        q_values = agent(state_tensor)  # Shape: (1, max_detections, num_actions_per_detection)
        q_values = q_values.squeeze(0)  # Shape: (max_detections, num_actions_per_detection)

        # Select actions with highest Q-value for each detection
        action = q_values.argmax(dim=1).cpu().numpy()

        # Take action
        next_state, reward, done, _ = env.step(action)

        # Store tracking results
        current_circle = env.unique_circles[env.current_circle_index - 1]
        for i, track_assignment in enumerate(action):
            if i >= len(state['detections']):
                break
            detection = state['detections'][i]
            if np.all(detection == 0):
                continue  # Ignore padding
            assigned_track = track_assignment
            if assigned_track < env.max_tracks:
                track_ids = list(env.tracks.keys())
                if assigned_track >= len(track_ids):
                    continue  # Invalid track assignment

                track_id = track_ids[assigned_track]
                if track_id not in tracks_output:
                    tracks_output[track_id] = []
                tracks_output[track_id].append({
                    'Time': detection[4],
                    'SlantRange': np.linalg.norm(detection[:3]),
                    'Azimuth': np.degrees(np.arctan2(detection[1], detection[0])),
                    'Elevation': np.degrees(np.arcsin(detection[2] / np.linalg.norm(detection[:3]))),
                    'RadialVelocity': detection[3],
                    'X': detection[0],
                    'Y': detection[1],
                    'Z': detection[2],
                    'Group_Size': 0,
                    'Circle': current_circle,
                    'Track_ID': track_id
                })

        state = next_state

    # Output the tracking results into a csv file
    tracks_df = pd.DataFrame()
    for track_id, track_data in tracks_output.items():
        track_df = pd.DataFrame(track_data)
        tracks_df = pd.concat([tracks_df, track_df])

    # sort the dataframe by time
    tracks_df = tracks_df.sort_values(by='Time')

    tracks_df.to_csv('./tracks_output.csv', index=False)

# ==========================
# Main Execution
# ==========================

if __name__ == '__main__':
    # Load data
    data = load_data('../../data/event_1/raw_tracks_1.csv')

    # Initialize environment
    env = RadarTrackingEnv(data)

    # State and action sizes
    state_size = (env.max_detections + env.max_tracks) * 5  # Each detection and track has 5 features
    action_size = env.max_tracks + 1  # For each detection, assign to a track or start a new one

    # Initialize agent
    agent = DQNAgent(detection_size=env.max_detections, track_size=env.max_tracks, action_size=action_size).to(device)

    # Train the agent
    train_agent(env, agent, episodes=300, tolerance=50)  # You can increase the number of episodes

    # Evaluate the agent
    evaluate_agent(env, agent)

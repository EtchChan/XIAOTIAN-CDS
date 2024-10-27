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

# ==========================
# Data Preprocessing
# ==========================

def load_data(filepath):
    # Load the data
    data = pd.read_csv(filepath)

    # Convert column names to English for ease of use
    data.columns = ['Time', 'SlantRange', 'Azimuth', 'Elevation', 'RadialVelocity', 'Circle']

    return data

# ==========================
# Environment Definition
# ==========================

class RadarTrackingEnv(gym.Env):
    def __init__(self, data):
        super(RadarTrackingEnv, self).__init__()

        self.data = data
        self.unique_times = self.data['Time'].unique()
        self.current_time_index = 0
        self.max_time_index = len(self.unique_times) - 1

        # Define maximum number of tracks and detections
        self.max_tracks = 10
        self.max_detections = 20

        # Define action and observation spaces
        # Action space: For each detection, assign it to a track or start a new track
        self.action_space = spaces.MultiDiscrete([self.max_tracks + 1] * self.max_detections)

        # Observation space: Positions and velocities of detections and tracks
        self.observation_space = spaces.Dict({
            'detections': spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_detections, 5), dtype=np.float32),
            'tracks': spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_tracks, 5), dtype=np.float32),
        })

        # Initialize state variables
        self.tracks = {}  # Track ID --> Track State
        self.track_id_counter = 0

    def reset(self):
        self.current_time_index = 0
        self.tracks = {}
        self.track_id_counter = 0

        return self._get_observation()

    def _get_observation(self):
        current_time = self.unique_times[self.current_time_index]
        detections = self.data[self.data['Time'] == current_time]
        detection_states = detections[['SlantRange', 'Azimuth', 'Elevation', 'RadialVelocity', 'Circle']].values

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
        current_time = self.unique_times[self.current_time_index]
        detections = self.data[self.data['Time'] == current_time]
        detection_states = detections[['SlantRange', 'Azimuth', 'Elevation', 'RadialVelocity', 'Circle']].values

        # Apply action
        reward = self._compute_reward(action, detection_states)

        # Update tracks based on action
        self._update_tracks(action, detection_states)

        # Move to next time step
        self.current_time_index += 1
        done = self.current_time_index >= self.max_time_index

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
            assigned_track = track_assignment

            # If assigned to an existing track
            if assigned_track < len(self.tracks):
                track_ids = list(self.tracks.keys())
                if assigned_track >= len(track_ids):
                    continue  # Invalid track assignment

                track_id = track_ids[assigned_track]
                previous_state = self.tracks[track_id]

                # Compute predicted state (simple linear prediction)
                predicted_state = previous_state  # For simplicity; can use motion model

                # Compute distance between detection and predicted track state
                position_distance = np.linalg.norm(detection[:3] - predicted_state[:3])
                velocity_difference = np.abs(detection[3] - predicted_state[3])

                # Compute reward
                reward -= position_distance  # Reward closer detections
                reward -= velocity_difference * 0.5  # Penalize large velocity changes

                # Penalize unphysical movements
                if position_distance > 100:  # Distance threshold
                    reward -= 20
                if velocity_difference > 10:  # Velocity threshold
                    reward -= 10
            else:
                # Start a new track
                # Penalize creating too many new tracks
                reward -= 5

        # Small positive reward for each maintained track
        reward += len(self.tracks) * 0.1

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
            if assigned_track < len(self.tracks):
                track_ids = list(self.tracks.keys())
                if assigned_track >= len(track_ids):
                    continue  # Invalid track assignment

                track_id = track_ids[assigned_track]
                # Update track state with the new detection
                updated_tracks[track_id] = detection
            else:
                # Start a new track
                track_id = self.track_id_counter
                self.track_id_counter += 1
                updated_tracks[track_id] = detection

        # Update tracks with the new states
        self.tracks = updated_tracks

# ==========================
# Agent Definition with Multi-Head Attention
# ==========================

class DQNAgent(nn.Module):
    def __init__(self, state_size, action_size, n_heads=4):
        super(DQNAgent, self).__init__()

        self.detection_input_dim = state_size // 2
        self.track_input_dim = state_size // 2
        self.embedding_dim = 128

        # Embedding layers
        self.detection_embedding = nn.Linear(self.detection_input_dim, self.embedding_dim)
        self.track_embedding = nn.Linear(self.track_input_dim, self.embedding_dim)

        # Multi-Head Attention
        self.multihead_attention = nn.MultiheadAttention(embed_dim=self.embedding_dim, num_heads=n_heads, batch_first=True)

        # Fully connected layers after attention
        self.fc1 = nn.Linear(self.embedding_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.out = nn.Linear(128, action_size *  self.action_per_detection())

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

def train_agent(env, agent, episodes=50):
    optimizer = optim.Adam(agent.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    replay_buffer = deque(maxlen=5000)
    batch_size = 32
    gamma = 0.99
    epsilon = 1.0  # Exploration rate
    epsilon_decay = 0.995
    epsilon_min = 0.01

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Epsilon-greedy action selection
            if np.random.rand() <= epsilon:
                # Random action
                action = np.random.randint(0, env.max_tracks + 1, size=env.max_detections)
            else:
                # Use the DQN to get action
                # Prepare state tensors
                detections = torch.FloatTensor(state['detections']).unsqueeze(0)  # Shape: (1, max_detections, 5)
                tracks = torch.FloatTensor(state['tracks']).unsqueeze(0)  # Shape: (1, max_tracks, 5)
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
                state_detections = torch.FloatTensor([s['detections'] for s in states_batch])
                state_tracks = torch.FloatTensor([s['tracks'] for s in states_batch])
                next_state_detections = torch.FloatTensor([ns['detections'] if ns is not None else np.zeros_like(s['detections']) for s, ns in zip(states_batch, next_states_batch)])
                next_state_tracks = torch.FloatTensor([ns['tracks'] if ns is not None else np.zeros_like(s['tracks']) for s, ns in zip(states_batch, next_states_batch)])

                actions_tensor = torch.LongTensor(actions_batch)  # Shape: (batch_size, max_detections)
                rewards_tensor = torch.FloatTensor(rewards_batch)
                dones_tensor = torch.FloatTensor(dones_batch)

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

        print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.4f}")

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
        state_tensor = {'detections': detections, 'tracks': tracks}

        q_values = agent(state_tensor)  # Shape: (1, max_detections, num_actions_per_detection)
        q_values = q_values.squeeze(0)  # Shape: (max_detections, num_actions_per_detection)

        # Select actions with highest Q-value for each detection
        action = q_values.argmax(dim=1).cpu().numpy()

        # Take action
        next_state, reward, done, _ = env.step(action)

        # Store tracking results
        current_time = env.unique_times[env.current_time_index - 1]
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
                    'Time': current_time,
                    'SlantRange': detection[0],
                    'Azimuth': detection[1],
                    'Elevation': detection[2],
                    'RadialVelocity': detection[3],
                    'Circle': detection[4]
                })

        state = next_state

    # Output the tracking results
    print("Tracking Results:")
    for track_id, observations in tracks_output.items():
        print(f"Track ID: {track_id}")
        for obs in observations:
            print(obs)
        print("----")

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
    agent = DQNAgent(state_size=state_size, action_size=action_size)

    # Train the agent
    train_agent(env, agent, episodes=10)

    # Evaluate the agent
    evaluate_agent(env, agent)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, namedtuple


# Define the DQN model
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Define the Multi-Agent DQN system
class MADQNDroneTracker:
    def __init__(self, num_agents, state_size, action_size, radar_size):
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.radar_size = radar_size

        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        self.agents = [DQN(state_size, action_size) for _ in range(num_agents)]
        self.target_agents = [DQN(state_size, action_size) for _ in range(num_agents)]
        self.optimizers = [optim.Adam(agent.parameters(), lr=self.learning_rate) for agent in self.agents]

        self.update_target_networks()

    def update_target_networks(self):
        for agent, target_agent in zip(self.agents, self.target_agents):
            target_agent.load_state_dict(agent.state_dict())

    def get_action(self, state, agent_index):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            q_values = self.agents[agent_index](state)
            return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        for agent_index in range(self.num_agents):
            agent_batch = [transition[agent_index] for transition in minibatch]
            states, actions, rewards, next_states, dones = zip(*agent_batch)

            states = torch.stack(states)
            next_states = torch.stack(next_states)
            actions = torch.tensor(actions)
            rewards = torch.tensor(rewards)
            dones = torch.tensor(dones, dtype=torch.float32)

            current_q_values = self.agents[agent_index](states).gather(1, actions.unsqueeze(1))
            next_q_values = self.target_agents[agent_index](next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

            loss = nn.MSELoss()(current_q_values, target_q_values.unsqueeze(1))

            self.optimizers[agent_index].zero_grad()
            loss.backward()
            self.optimizers[agent_index].step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, num_episodes, radar_data_generator, confidence_func, threshold):
        for episode in range(num_episodes):
            radar_data = radar_data_generator()
            state = torch.zeros(self.num_agents, self.state_size)
            total_reward = 0

            for step in range(100):  # Assume 100 steps per episode
                actions = []
                for agent_index in range(self.num_agents):
                    action = self.get_action(state[agent_index], agent_index)
                    actions.append(action)

                next_radar_data = radar_data_generator()
                next_state = torch.zeros(self.num_agents, self.state_size)
                reward = self.calculate_reward(actions, radar_data, confidence_func, threshold)
                total_reward += reward

                for agent_index in range(self.num_agents):
                    if actions[agent_index] > 0:
                        next_state[agent_index] = radar_data[actions[agent_index] - 1]

                done = (step == 99)  # Episode ends after 100 steps

                self.remember([state[i] for i in range(self.num_agents)],
                              actions,
                              [reward] * self.num_agents,
                              [next_state[i] for i in range(self.num_agents)],
                              [done] * self.num_agents)

                state = next_state
                radar_data = next_radar_data

                self.replay()

            if episode % 10 == 0:
                print(f"Episode {episode}, Total Reward: {total_reward}")

            if episode % 100 == 0:
                self.update_target_networks()

    def calculate_reward(self, actions, radar_data, confidence_func, threshold):
        reward = 0
        selected_points = [radar_data[action - 1] if action > 0 else None for action in actions]

        # Check for duplicate selections
        if len(set(actions)) < len(actions):
            reward -= 10 * self.num_agents

        # Calculate confidence-based reward
        for i in range(len(selected_points)):
            if selected_points[i] is not None:
                for j in range(i + 1, len(selected_points)):
                    if selected_points[j] is not None:
                        confidence = confidence_func(selected_points[i], selected_points[j])
                        reward += confidence - threshold

        return reward


# Example usage
num_agents = 5
state_size = 32  # Dimension of each radar point
action_size = 11  # 0-10, where 0 means no selection and 1-10 correspond to radar points
radar_size = 10  # Number of radar points in each step

ma_dqn = MADQNDroneTracker(num_agents, state_size, action_size, radar_size)


# Define a simple radar data generator
def radar_data_generator():
    return torch.randn(radar_size, state_size)


# Define a simple confidence function
def confidence_func(point1, point2):
    return torch.exp(-torch.norm(point1 - point2)).item()


# Train the system
ma_dqn.train(num_episodes=1000, radar_data_generator=radar_data_generator,
             confidence_func=confidence_func, threshold=0.5)

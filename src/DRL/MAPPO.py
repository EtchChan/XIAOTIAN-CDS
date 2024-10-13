import numpy as np
"""
Brief: This script implement a multi-agent policy proximal optimization (MAPPO) algorithm using Ray RLlib.
       It is used to solve a multi-agent drone tracking problem, where multiple agents need to decide whether to
       accept a radar point based on their own estimates of the target state.

Author: CHEN Yi-xuan

updateDate: 2024-10-13
"""
import gymnasium as gym
from gymnasium import spaces
from ray.rllib.algorithms import ppo
from ray import tune
import ray
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from scipy.stats import chi2
import numpy as np
from ray.tune.registry import register_env
from ray.rllib.policy.policy import PolicySpec


def load_radar_data(file_path):
    radar_data = np.load(file_path)
    return radar_data


class DroneTrackingEnv(MultiAgentEnv):
    def __init__(self, config):
        super().__init__()
        self.agent_states = None
        self.current_step = 0
        self.num_agents = config.get('num_agents', 3)
        self.agents = ["agent_{}".format(i) for i in range(self.num_agents)]
        self.threshold = config.get('threshold', 0.5)
        self.max_accept_steps = config.get('max_accept_steps', float('inf'))  # Configurable upper bound
        self.radar_data = config.get('radar_data')
        if self.radar_data is None:
            raise ValueError("radar_data must be provided in config.")
        self.num_steps = len(self.radar_data)
        self.action_space = spaces.Discrete(2)
        # Observation: [time, x, y, z, v_radial_x, v_radial_y, v_radial_z,
        # x_est, y_est, z_est, v_x_est, v_y_est, v_z_est, confidence]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32)
        self.reset()

    def reset(self, *, seed=None, options=None):
        self.current_step = 0
        self.agent_states = {
            agent: {
                'x_est': 0.0,
                'y_est': 0.0,
                'z_est': 0.0,
                'v_x_est': 0.0,
                'v_y_est': 0.0,
                'v_z_est': 0.0,
                'P': np.eye(6),  # Covariance matrix
                'confidence': 1.0,
                'accept_count': 0  # Initialize accept count
            }
            for agent in self.agents
        }
        self.agent_tracks = {agent: [] for agent in self.agents}

        # Return initial observations
        observations = self._get_observations()
        return observations

    def step(self, action_dict):
        rewards = {}
        observations = {}
        infos = {}
        dones = {}

        radar_point = self.radar_data[self.current_step]

        for agent in self.agents:
            action = action_dict.get(agent, 0)
            state = self.agent_states[agent]

            # Check if max_accept_steps is reached
            if state['accept_count'] >= self.max_accept_steps:
                # Agent cannot accept more; force action to 0
                if action == 1:
                    action = 0
                    # Optionally, inform the agent
                    infos[agent] = {'max_accept_steps_reached': True}

            if action == 1:
                # Agent accepts the radar point
                measurement = radar_point[1:7]  # x, y, z, v_radial_x, v_radial_y, v_radial_z
                # Run Kalman filter update
                state = self.kalman_filter_update(state, measurement)
                self.agent_states[agent] = state
                self.agent_tracks[agent].append(self.current_step)
                state['accept_count'] += 1  # Increment accept count
                # Calculate reward
                confidence = state['confidence']
                reward = confidence - self.threshold
            else:
                # Agent does not accept the point
                reward = 0.0

            rewards[agent] = reward
            observations[agent] = self._get_observation(agent)
            infos.setdefault(agent, {})
            dones[agent] = False  # We'll handle 'done' after the loop

        # Negative reward if two agents have identical non-zero tracks
        for i in range(len(self.agents)):
            for j in range(i + 1, len(self.agents)):
                agent_i = self.agents[i]
                agent_j = self.agents[j]
                track_i = self.agent_tracks[agent_i]
                track_j = self.agent_tracks[agent_j]

                if len(track_i) > 0 and len(track_j) > 0 and track_i == track_j:
                    penalty = -10 * self.num_agents
                    rewards[agent_i] += penalty
                    rewards[agent_j] += penalty

        self.current_step += 1
        done = self.current_step >= self.num_steps

        for agent in self.agents:
            dones[agent] = done
        dones["__all__"] = done

        return observations, rewards, dones, infos

    def kalman_filter_update(self, state, measurement):
        # Kalman filter parameters
        dt = 0.1  # Time step
        A = np.eye(6)
        A[0, 3] = dt
        A[1, 4] = dt
        A[2, 5] = dt
        H = np.eye(6)
        Q = np.eye(6) * 0.1  # Process noise covariance
        R = np.eye(6) * 1.0  # Measurement noise covariance

        # Prior state estimate
        x_prior = np.array([
            state['x_est'],
            state['y_est'],
            state['z_est'],
            state['v_x_est'],
            state['v_y_est'],
            state['v_z_est']
        ])

        P_prior = state['P']

        # Prediction
        x_pred = A @ x_prior
        P_pred = A @ P_prior @ A.T + Q

        # Innovation
        y = measurement - H @ x_pred
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)

        # Update
        x_post = x_pred + K @ y
        P_post = (np.eye(6) - K @ H) @ P_pred

        # Confidence calculation (assuming chi-square distribution of innovation)
        error = y.T @ np.linalg.inv(S) @ y
        confidence = 1 - chi2.cdf(error, df=6)

        # Update state
        state['x_est'] = x_post[0]
        state['y_est'] = x_post[1]
        state['z_est'] = x_post[2]
        state['v_x_est'] = x_post[3]
        state['v_y_est'] = x_post[4]
        state['v_z_est'] = x_post[5]
        state['P'] = P_post
        state['confidence'] = confidence

        return state

    def _get_observations(self):
        observations = {}
        for agent in self.agents:
            observations[agent] = self._get_observation(agent)
        return observations

    def _get_observation(self, agent):
        radar_point = self.radar_data[self.current_step]
        state = self.agent_states[agent]
        obs = np.concatenate((
            radar_point[0:7],  # time, x, y, z, v_radial_x, v_radial_y, v_radial_z
            np.array([
                state['x_est'],
                state['y_est'],
                state['z_est'],
                state['v_x_est'],
                state['v_y_est'],
                state['v_z_est'],
                state['confidence']
            ])
        )).astype(np.float32)
        return obs


def create_environment(env_config):
    def env_creator(env_config):
        return DroneTrackingEnv(env_config)
    register_env("DroneTrackingEnv", env_creator)
    return "DroneTrackingEnv"


def configure_policies(env_config):
    policies = {}
    for i in range(env_config['num_agents']):
        agent_id = f"agent_{i}"
        policies[agent_id] = PolicySpec(
            policy_class=None,  # Use default policy
            observation_space=None,  # Use env's observation space
            action_space=None,  # Use env's action space
            config={}
        )
    return policies


def train_agent(env_name, env_config, policies):
    # Prepare config dict
    config = {
        'env': env_name,
        'env_config': env_config,
        'multiagent': {
            'policies': policies,
            'policy_mapping_fn': lambda agent_id, episode, **kwargs: agent_id,
        },
        'framework': 'torch',
        'num_gpus': 1,
        'num_workers': 10,
        'log_level': 'WARN',
    }
    stop = {'training_iteration': 20}  # Adjust as needed
    results = tune.run(
        'PPO',
        config=config,
        stop=stop,
        verbose=1,
        checkpoint_at_end=True
    )
    return results


def evaluate_agent(env_config, policies, checkpoint_path):
    # Create the agent using the checkpoint
    config = {
        'env': 'DroneTrackingEnv',
        'env_config': env_config,
        'multiagent': {
            'policies': policies,
            'policy_mapping_fn': lambda agent_id, episode, **kwargs: agent_id,
        },
        'framework': 'torch',
        'num_gpus': 0,
        'num_workers': 0,
        'log_level': 'WARN',
    }
    agent = ppo.PPOTrainer(config=config)
    agent.restore(checkpoint_path)
    # Evaluate the trained policy
    env = DroneTrackingEnv(env_config)
    observations = env.reset()
    done = False
    total_rewards = {agent_id: 0 for agent_id in env.agents}
    while not done:
        action_dict = {}
        for agent_id, obs in observations.items():
            action = agent.compute_single_action(obs, policy_id=agent_id)
            action_dict[agent_id] = action
        observations, rewards, dones, infos = env.step(action_dict)
        done = dones["__all__"]
        for agent_id, reward in rewards.items():
            total_rewards[agent_id] += reward
    print("Total rewards:", total_rewards)


def tune_policy():
    # Initialize Ray
    ray.shutdown()
    ray.init()
    # Load radar data
    radar_data = load_radar_data('../../data/event_1/raw_tracks_1.npy')
    # Environment configuration
    env_config = {
        'num_agents': 10,  # Number of agents (M)
        'threshold': 0.3,
        'max_accept_steps': 309,  # Configurable upper bound on accept actions
        'radar_data': radar_data,  # Provide radar data here
    }
    # Create environment
    env_name = create_environment(env_config)
    # Configure policies
    policies = configure_policies(env_config)
    # Train agent
    results = train_agent(env_name, env_config, policies)
    # Get best checkpoint
    best_trial = results.get_best_trial("episode_reward_mean", mode="max")
    best_checkpoint = results.get_best_checkpoint(best_trial, metric="episode_reward_mean", mode="max")
    checkpoint_path = best_checkpoint
    # Evaluate agent
    evaluate_agent(env_config, policies, checkpoint_path)
    # Shutdown Ray
    ray.shutdown()


if __name__ == '__main__':
    tune_policy()

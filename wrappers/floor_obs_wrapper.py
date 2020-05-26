import gym
import numpy as np

class FloorObsWrapper(gym.Wrapper): # Floor and int all observations
    def __init__(self, gym_env, scaling_factor=1):
        env = gym_env
        super(FloorObsWrapper, self).__init__(env)
        self.scaling_factor = scaling_factor

    def step(self, action):
        """Wrap the step function to tokenize observations"""
        observation, reward, done, info = super().step(action)
        return self.floor_obs(observation), reward, done, info

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        return self.floor_obs(observations)

    def floor_obs(self,observation):
        return np.floor(observation*self.scaling_factor)

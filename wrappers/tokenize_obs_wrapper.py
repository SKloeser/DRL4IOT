import gym
import numpy as np
import nltk

class TokenizeObsWrapper(gym.Wrapper):
    def __init__(self, gym_env):
        env = gym_env
        super(TokenizeObsWrapper, self).__init__(env)

    def step(self, action):
        """Wrap the step function to tokenize observations"""
        observation, reward, done, info = super().step(action)
        return self.tokenize_obs(observation), reward, done, info

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        return self.tokenize_obs(observations)

    def tokenize_obs(self,observation):
        tokenized = nltk.word_tokenize(str(observation))
        return tokenized

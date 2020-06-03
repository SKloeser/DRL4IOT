import gym
import numpy as np
import os

class GloveObsWrapper(gym.Wrapper):
    def __init__(self, gym_env, glove_model_path):
        env = gym_env
        self.glove = self.loadGloveModel(glove_model_path)
        super(GloveObsWrapper, self).__init__(env)
        print("shape", env.observation_space.shape) 
        high = np.ones(len(self.unwrapped.observation_space.sample()) * len(self.glove["0"])) * 100
        self.observation_space = gym.spaces.Box(-high, high)
        # Adapt the observation_space to the observations.
        # the observation space has still the shape for the unrapped observations: 4 ints
        # reset the environment to get a dummy observation for the first dimension
        # use entry "0" in the glove-model to get the second dimension
        obs_dummy = env.reset()
        self.observation_space.shape = (len(obs_dummy), len(self.glove["0"]))
        self.unwrapped.observation_space = gym.spaces.Box(-high, high)

    def step(self, action):
        """Wrap the step functions to embed obs using GloVe"""
        observation, reward, done, info = super().step(action)
        return self.embed_obs(observation), reward, done, info

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        return self.embed_obs(observations)

    def embed_obs(self, observation):
        def isfloat(value):
            try:
                float(value)
                return True
            except ValueError:
                return False
        new_observation = []
        for obs in range(len(observation)):
            if not isfloat(observation[obs]):
                try:
                    new_observation.append(self.glove[str(observation[obs])])
                except KeyError:
                    raise KeyError("String '" + str(observation[obs]) + "' not found in GloVe set")
                continue
            new_observation.append(self.glove[str((np.clip(int(float(observation[obs])), -75, 75)))])
        return np.array(new_observation)

    def loadGloveModel(self, gloveFile):
        f = open(os.path.join(os.getcwd(),gloveFile), 'r' ,encoding="utf8")
        model = {}
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word] = embedding
        return model

import numpy as np
import torch

class UnityEnv():
    """Unity Reacher Environment Wrapper
        https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md

    """
    def __init__(self, env_file='data/Reacher.exe', no_graphics=True, mlagents=False):
        if mlagents:
            from mlagents.envs.environment import UnityEnvironment
        else:
            from unityagents import UnityEnvironment
        self.env = UnityEnvironment(file_name=env_file, no_graphics=no_graphics)  
        self.brain_name = self.env.brain_names[0]
        brain = self.env.brains[self.brain_name]
        self.action_size = brain.vector_action_space_size
        if type(self.action_size) != int:
            self.action_size = self.action_size[0]
        env_info = self.env.reset(train_mode=True)[self.brain_name] 
        self.state_size = env_info.vector_observations.shape[1]
        self.num_agents = len(env_info.agents)
        
    def reset(self, train=True):
        env_info = self.env.reset(train_mode=train)[self.brain_name] 
        return env_info.vector_observations

    def close(self):
        self.env.close()

    def step(self, actions):
        actions = np.clip(actions, -1, 1) 
        env_info = self.env.step(actions)[self.brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
        return next_states, np.array(rewards), np.array(dones)

    @property
    def action_shape(self):
        return (self.num_agents, self.action_size)

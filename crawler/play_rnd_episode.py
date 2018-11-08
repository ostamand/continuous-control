from mlagents.envs.environment import UnityEnvironment
import pdb
import numpy as np
import os

def play():
    env = UnityEnvironment(file_name=os.path.join('data', 'Crawler_Windows_x86_64.exe'), no_graphics=False)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    action_size = brain.vector_action_space_size[0]

    score = np.zeros(12)
    env.reset(train_mode=False)[brain_name]
    for i in range(200):
        action = (np.random.rand(12,action_size)-0.5)*2
        env_info = env.step(action)[brain_name]
        #state = env_info.vector_observations
        reward = env_info.rewards  
        done = env_info.local_done 
        score += reward
        if np.array(done).any():
            break
    print(f'Average score: {np.mean(reward):.2f}')
    print(f'{i} timesteps')

    env.close()

if __name__ == '__main__':
    play()
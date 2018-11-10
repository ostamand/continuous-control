import argparse
import torch
import numpy as np
from unity_env import UnityEnv

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def watch_rnd_game(steps):
    env = UnityEnv(env_file='data/Crawler/Crawler_Windows_x86_64.exe',
                   no_graphics=False, mlagents=True)
    env.reset(train=False)

    rewards = np.zeros(env.num_agents)
    for i in range(steps):
        action = np.random.rand(env.num_agents, env.action_size)
        _, r, done = env.step(action)
        rewards += r
        if done.all():
            break
    print(f'Average score of 20 agents is: {np.mean(rewards):.2f}')
    env.close()

#pylint: disable=invalid-name
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch a rnd Crawler agent")
    parser.add_argument('--steps', '-s', default=500)
    args = parser.parse_args()
    watch_rnd_game(int(args.steps))

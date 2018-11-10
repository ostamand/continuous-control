import argparse
import numpy as np
import torch
from unity_env import UnityEnv
from model import ReacherActorCritic, CrawlerActorCritic

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def watch_agent(env_name, agent_ckpt):
    device = torch.device(DEVICE)

    if env_name == 'reacher':
        env = UnityEnv(env_file='data/Reacher.exe', no_graphics=False)
        policy = ReacherActorCritic(env.state_size, env.action_size).to(device)
    else:
        env = UnityEnv(env_file='data/Crawler/Crawler_Windows_x86_64.exe', no_graphics=False)
        policy = CrawlerActorCritic(env.state_size, env.action_size).to(device)

    checkpoint = torch.load(agent_ckpt, map_location=DEVICE)
    policy.load_state_dict(checkpoint)

    rewards = np.zeros(env.num_agents)
    state = env.reset(train=False)
    while True:
        action, _, _, _ = policy(torch.from_numpy(state).float().to(device))
        state, r, done = env.step(action.detach().cpu().numpy())
        rewards += r
        if done.all():
            break
    env.close()
    print(f'Average score of 20 agents is: {np.mean(rewards):.2f}')

if __name__ == "__main__":
    #pylint: disable=invalid-name
    parser = argparse.ArgumentParser(description='Watch Reacher trained agent') 
    parser.add_argument("--agent", "-a", help="agent to watch", default='saved_models/ppo.ckpt')
    parser.add_argument("--env", "-e", help="Unity environment to run", default='reacher')
    args = parser.parse_args()

    watch_agent(args.env, args.agent)

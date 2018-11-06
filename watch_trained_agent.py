from reacher_env import ReacherEnvironment
from model import GaussianActorCritic
import argparse
import torch 
import numpy as np

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def watch_agent(agent_ckpt):
    device = torch.device(DEVICE)
    env = ReacherEnvironment(no_graphics=False)
    
    policy = GaussianActorCritic(env.state_size, env.action_size).to(device)
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
    parser = argparse.ArgumentParser(description = 'Watch Reacher trained agent') 
    parser.add_argument("--agent", "-a", help="agent to watch", default='saved_models/ppo.ckpt')
    args = parser.parse_args()

    watch_agent(args.agent)
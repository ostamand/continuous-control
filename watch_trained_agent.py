import argparse
import numpy as np
import torch
from unity_env import UnityEnv
from model import ReacherActorCritic, CrawlerActorCritic

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def watch_agent(env_name, agent_ckpt, steps):
    device = torch.device(DEVICE)

    if env_name == 'reacher':
        env = UnityEnv(env_file='data/Reacher.exe', no_graphics=False)
        policy = ReacherActorCritic(env.state_size, env.action_size).to(device)
    else:
        env = UnityEnv(env_file='data/Crawler/Crawler_Windows_x86_64.exe', no_graphics=False, mlagents=True)
        policy = CrawlerActorCritic(env.state_size, env.action_size).to(device)

    checkpoint = torch.load(agent_ckpt, map_location=DEVICE)
    policy.load_state_dict(checkpoint)

    running_rewards = np.zeros(env.num_agents)
    scores = np.zeros(env.num_agents)
    state = env.reset(train=False)
    for step_i in range(steps):
        action, _, _, _ = policy(torch.from_numpy(state).float().to(device))
        state, r, done = env.step(action.detach().cpu().numpy())
        running_rewards += r

        # check if agent is done
        agents_are_done = True
        for i in range(env.num_agents):
            if done[i] and scores[i] == 0:
                scores[i] = running_rewards[i]
            if scores[i] == 0:
                agents_are_done = False
        if agents_are_done:
            break

    env.close()
    print(f'Average score of 20 agents is: {np.mean(scores):.2f}')

if __name__ == "__main__":
    #pylint: disable=invalid-name
    parser = argparse.ArgumentParser(description='Watch Reacher trained agent')
    parser.add_argument("--agent", "-a", help="agent to watch", default='saved_models/ppo.ckpt')
    parser.add_argument("--env", "-e", help="Unity environment to run", default='reacher')
    parser.add_argument("--steps", "-s", help="Number of steps to run per agents", default=1000)
    args = parser.parse_args()

    watch_agent(args.env, args.agent, args.steps)

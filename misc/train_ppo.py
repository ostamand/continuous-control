import numpy as np
import pickle

def train(agent, 
    iterations=1000, 
    log_each=10, 
    solved=90, 
    decay_steps=None, 
    out_file=None
    ):
    rewards = []
    beta = agent.beta
    last_saved = 0

    for it in range(iterations):
        if decay_steps:
            ratio = 1 - (agent.steps / decay_steps)
            agent.beta = max(ratio * beta, 0)

        agent.step()

        if len(agent.episodes_reward) >= 100:
            r = agent.episodes_reward[:-101:-1]
            rewards.append((agent.steps, min(r), max(r), np.mean(r), np.std(r)))

        if (it+1) % log_each == 0:
            summary = ''
            #pylint: disable=line-too-long
            if rewards:
                mean = rewards[-1][3]
                summary = f', Rewards: {mean:.2f}/{rewards[-1][4]:.2f}/{rewards[-1][1]:.2f}/{rewards[-1][2]:.2f} mean/std/min/max'
                if out_file and mean >= solved and mean > last_saved:
                    last_saved = mean
                    agent.save(out_file)
                    summary += " (saved)"
            print(f"Iteration: {it+1:d}, Episodes: {len(agent.episodes_reward)}, Steps: {agent.steps:d}, Beta: {agent.beta:.3f}, Clip: {agent.ratio_clip:.3f}{summary}")

    pickle.dump(rewards,open('rewards.p','wb'))
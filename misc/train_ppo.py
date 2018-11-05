import numpy as np

def train(agent, iterations=1000, log_each=10, solved=90, decay_steps=None):
    rewards = []
    stop = False
    beta = agent.beta

    for it in range(iterations):

        if decay_steps:
            ratio = 1 - (agent.steps / decay_steps)
            agent.beta = max(ratio * beta, 0)

        agent.step()

        if len(agent.episodes_reward) >= 100:
            r = agent.episodes_reward[:-101:-1]
            rewards.append((agent.steps, min(r), max(r), np.mean(r), np.std(r)))

        summary = ''
        if (it+1) % log_each == 0:

            #pylint: disable=line-too-long
            if rewards:
                summary = f', Rewards: {rewards[-1][3]:.2f}/{rewards[-1][4]:.2f}/{rewards[-1][1]:.2f}/{rewards[-1][2]:.2f} mean/std/min/max'

                if rewards[-1][3] >= solved:
                    summary += ' Solved.'
                    stop = True

            print(f"Iteration: {it+1:d}, Episodes: {len(agent.episodes_reward)}, Steps: {agent.steps:d}, Beta: {agent.beta:.3f}, Clip: {agent.ratio_clip:.3f}{summary}")

        if stop:
            break

from misc.train_ppo import train
from reacher_env import ReacherEnvironment
from agent_ppo import Agent

if __name__ == '__main__':
    #pylint: disable=invalid-name
    iterations = 1000
    #max_steps = 200
    parallel_envs = 12
    gamma = 0.99
    timesteps = 128
    ratio_clip = 0.2
    batch_size = int(32*parallel_envs)
    epochs = 10
    gradient_clip = 5
    lrate = 1e-3
    log_each = 10
    beta = 0.0
    gae_tau = 0.95
    decay_steps = None
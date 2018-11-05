from misc.train_ppo import train
from reacher_env import ReacherEnvironment
from agent_ppo import Agent
from model import GaussianActorCritic
import torch

if __name__ == '__main__':
    #pylint: disable=invalid-name
    iterations = 1000
    gamma = 0.99
    timesteps = 100
    ratio_clip = 0.2
    batch_size = int(32*20)
    epochs = 10
    gradient_clip = 10.0
    lrate = 1e-4
    log_each = 10
    beta = 0.01
    gae_tau = 0.95
    decay_steps = None
    solved = 30.0

    # gamma=0.99
    # timesteps=100
    # ratio_clip=0.2
    # batch_size=32*20
    # epochs=10
    # gradient_clip=10.0
    # lrate=1e-3
    # beta=0.01
    # gae_tau=0.95

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    env = ReacherEnvironment()
    policy = GaussianActorCritic(env.state_size, env.action_size).to(device)
    a = Agent(
        env,
        policy,
        timesteps=timesteps,
        gamma=gamma,
        epochs=epochs,
        batch_size=batch_size,
        ratio_clip=ratio_clip,
        lrate=lrate,
        gradient_clip=gradient_clip,
        beta=beta,
        gae_tau=gae_tau
    )

    train(a, iterations=iterations, log_each=log_each, solved=solved, decay_steps=decay_steps )
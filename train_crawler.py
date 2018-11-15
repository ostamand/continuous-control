import torch
from misc.train_ppo import train
from unity_env import UnityEnv
from agent_ppo import Agent
from model import CrawlerActorCritic
from tensorboardX import SummaryWriter

if __name__ == '__main__':
    #pylint: disable=invalid-name
    writer = SummaryWriter()

    iterations = 10000
    gamma = 0.99
    nsteps = 2048
    ratio_clip = 0.2
    nbatchs = 32
    epochs = 10
    gradient_clip = 0.5
    lrate = 2e-4
    lrate_schedule = lambda it: max(0.995 ** it, 0.01)
    log_each = 1
    beta = 0.0
    gae_tau = 0.95
    decay_steps = None
    solved = 100.0
    weight_decay = 0
    out_file = 'saved_models/crawler_ppo.ckpt'
    restore = None

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    env = UnityEnv(env_file='data/Crawler/Crawler_Windows_x86_64.exe', mlagents=True)
    policy = CrawlerActorCritic(env.state_size, env.action_size).to(device)
    a = Agent(
        env,
        policy,
        nsteps=nsteps,
        gamma=gamma,
        epochs=epochs,
        nbatchs=nbatchs,
        ratio_clip=ratio_clip,
        lrate=lrate,
        lrate_schedule=lrate_schedule,
        weight_decay=weight_decay,
        gradient_clip=gradient_clip,
        beta=beta,
        gae_tau=gae_tau,
        restore=restore
    )

    train(a, iterations=iterations, log_each=log_each,
          solved=solved, out_file=out_file, writer=writer)

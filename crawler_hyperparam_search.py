#pylint: skip-file
import random
import os

import torch
from tensorboardX import SummaryWriter

from misc.train_ppo import train
from unity_env import UnityEnv
from agent_ppo import Agent
from model import CrawlerActorCritic

if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # parameters that are varied
    nbatchs = [16, 32, 64] # number of batch per training epoch
    lrates = [1e-4, 3e-4] # learning rate 
    clips = [0.1, 0.2] # propability ratio clipping
    nsteps = [512, 1024, 2048] # number of steps per agent per iteration
    epochs = [4, 10] # number of training epoch per iteration
    gae_taus =[0.95, 0.98] # GAE tau  
    lrate_decays = [0.995, 0.999, 0.9995]

    # parameters that are fixed 
    gamma = 0.99 # discount rate
    gradient_clip = 0.5 # gradient norm clipping
    log_each = 1 
    beta = 0.0 # entropy coefficient

    # number of hyperparam seach loop to run 
    search_loops = 100 

    # number of iterations per hyperparam loop
    iterations = 250

    # root logdir for tensorboard 
    root_logdir = 'search'

    for _ in range(search_loops):

        nbatch = random.choice(nbatchs)
        lrate = random.choice(lrates)
        clip = random.choice(clips)
        nstep = random.choice(nsteps)
        epoch = random.choice(epochs)
        gae_tau = random.choice(gae_taus)
        lrate_decay = random.choice(lrate_decays)
        lrate_schedule = lambda it: lrate_decay ** it

        summary = f'nbatch_{nbatch:d}_lrate_{lrate:.0E}_clip_{clip:.2f}'
        summary += f'_nstep_{nstep:d}_epoch_{epoch:d}_gae_{gae_tau:.2f}'
        summary += f'_lrdecay_{lrate_decay}'

        writer = SummaryWriter(os.path.join(root_logdir, summary))

        # create new environment
        env = UnityEnv(env_file='data/Crawler/Crawler_Windows_x86_64.exe', mlagents=True)

        # create new policy
        policy = CrawlerActorCritic(env.state_size, env.action_size).to(device)

        # create agent
        a = Agent(
            env,
            policy,
            nsteps=nstep,
            gamma=gamma,
            epochs=epoch,
            nbatchs=nbatch,
            ratio_clip=clip,
            lrate=lrate,
            gradient_clip=gradient_clip,
            beta=beta,
            gae_tau=gae_tau,
            lrate_schedule=lrate_schedule)

        # run training 
        print(f'Running: {summary}')
        train(a, iterations=iterations, log_each=log_each, writer=writer)

        # close env
        env.close()







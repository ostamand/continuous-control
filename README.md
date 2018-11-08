# Unity Reacher
![PPO agent](assets/unity_reacher_ppo_agent.gif)
This repository includes the code needed to train agents to solve Udacity version 2 (20 agents) navigation project. 

The environment is based on [Unity's Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md). It includes twenty separate agents controlling a double jointed arm. The goal of each agent is to move its hand to the target location (representated by a green sphere when the goal is met and a blueish sphere when it is not). 

Each timestep, the agent recieves:

-  A 28 element long vector representing the position, rotation, velocity and angular velocities of its two arm rigid bodies.
- A reward of +0.1 if the hand of the agent is within the target boundary.

The actions space is continuous and consists of the torque applicable to each joint (clipped between -1 and +1).

Finally, the environment is considered solved when the average total reward of the last 100 episodes over all parallel agents is greater than 30.0.

## Training 

To train the agent simply run `python train_reacher.py`. All hyperparameters can be modified within the script file.   

## Results 

A [trained model](saved_models/agent_ppo.ckpt) with an average score of 38.45 over 100 episodes of all 20 parallel agents is included in this repository.

For a more complete description of the results, refer to the [report](report.md) page.

To visualise the trained agent either follow this [link](https://youtu.be/ExtYVXhBvEI) or run:

```
python watch_trained_agent.py --agent data/ppo.ckpt
``` 

## Installation

Create a new Python 3.6 environment.

```
conda create --name reacher python=3.6 
activate reacher
```

Install ml-agents using the repository.

```
git clone https://github.com/Unity-Technologies/ml-agents.git
cd ml-agents
git checkout 0.4.0b
cd python 
pip install .
```

Install PyTorch using the recommended [pip command](https://pytorch.org/) from the PyTorch site. For example, to install with CUDA 9.2: 

```
conda install pytorch cuda92 -c pytorch
```

Clone this repository locally. 

```
git clone https://github.com/ostamand/continuous-control.git
```

Finally, download the environment which corresponds to your operationg system. Copy/paste the extracted content to the `data` subfolder. 

- [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip) 
- [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
- [Windows (32-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
- [Windows (64-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
# Unity Reacher
This repository includes the code needed to train agents to solve Udacity Version 2 (20 agents) navigation project. The environment is based on [Unity's Reacher environment](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md). 

## Training 

To train the agent simply run `python train.py`. All hyperparameters can be modified within the script file.   

## Results 

A [trained model]() with an average score of XX over 100 episodes of all 20 parallel agents is included in this repository.

For a more complete description of the results, refer to the [report](report.md) page.

To visualise the trained agent, use this [notebook]().

## Installation

Create a new Python 3.6 environment.

```
conda create --name drlnd python=3.6 
activate drlnd
```

Install ml-agents using the repository.

```
git clone https://github.com/Unity-Technologies/ml-agents.git
cd ml-agents
git checkout 0.4.0b
cd python 
pip install .
```

Install PyTorch using the recommended [pip command](https://pytorch.org/). For example, to install with CUDA 9.2: 

```
conda install pytorch cuda92 -c pytorch
```

Clone this repository locally. 

```
git clone https://github.com/ostamand/continuous-control.git
```

Finally, download the environment which correspond to your operationg system. Copy/paste the extracted content to the `data` subfolder. 

- [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip) 
- [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
- [Windows (32-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
- [Windows (64-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
# Report 

## Hyperparameters

The hyperparameters used during training are:

Parameter | Value | Description
------------ | ------------- | -------------
Parallel Agents | 20 | Number of agents trained simultaneously
Iterations | X | Number of iterations to run
Epochs | 10 | Number of training epoch per iteration
Batch size | 32*20 | Size of batch taken from the accumulated  trajectories
Timesteps | 500 | Number of steps per trajectory 
Gamma | 0.99 | Discount rate 
Ratio clip | 0.2 | Ratio used to clip r = new_probs/old_probs during training
Gradient clip | 10.0 | Maximum gradient norm 
Learning rate | 1e-4 | Learning rate 
Beta | 0.0 | Entropy coefficient 
Tau GAE | 0.95 |Generalized Advantage Estimate tau coefficient
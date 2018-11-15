# Crawler

![PPO Reacher](assets/unity_crawler_ppo_agent.gif)

## Training 

## Results 

A [trained model](saved_models/agent_ppo_crawler.ckpt) with an average score of 1358 over 100 episodes of all 20 parallel agents is included in this repository.

![PPO Reacher](assets/crawler_training.png)

To visualise the trained agent either follow this [link](https://youtu.be/cEbgqLacPgQ) or run:

```
python watch_trained_agent.py --agent data/agent_ppo_crawler.ckpt --env crawler
``` 


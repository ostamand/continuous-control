import matplotlib.pyplot as plt
import numpy as np
import pickle

if __name__ == "__main__":
    with open('rewards.p', 'rb') as f:
        data = pickle.load(f)
    steps, mins, maxs, means, stds = zip(*data)
    plt.plot(steps, means)
    plt.xlabel("Step")
    plt.ylabel("Average Score")
    plt.show()






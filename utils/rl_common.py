import numpy as np

def calculate_discounted_returns(rewards):
    """
    Calculate discounted reward and then normalize it
    See Sutton book for definition

    Params:
        rewards: list of rewards for every episode
    """
    returns = np.zeros(len(rewards))

    next_return = 0 # 0 because we start at the last timestep
    for t in reversed(range(0, len(rewards))):
        next_return = rewards[t] + args.gamma * next_return
        returns[t] = next_return
    # normalize for better statistical properties
    returns = (returns - returns.mean()) / (returns.std() + np.finfo(np.float32).eps)
    return returns


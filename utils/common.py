class CacheDict(dict):
    def __init__(self, *args):
        dict.__init__(self, args)

    def __getitem__(self, key):
        if dict.__contains__(self, key):
            val = dict.__getitem__(self, key)
        else:
            val = [] 
            dict.__setitem__(self, key, val)
        return val

    def __setitem__(self, key, val):
        dict.__setitem__(self, key, val)

class GradDict(dict):
    def __init__(self, *args):
        dict.__init__(self, args)

    def __getitem__(self, key):
        if dict.__contains__(self, key):
            val = dict.__getitem__(self, key)
        else:
            val = 0
            dict.__setitem__(self, key, val)
        return val

    def __setitem__(self, key, val):
        dict.__setitem__(self, key, val)


def softmax(x):
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    return probs


def mse_loss(pred, label):
    loss = np.mean(np.square(pred - label)) # just for logging. not actually needed for optimization
    dpred = 2*(pred - label)
    return loss, dpred

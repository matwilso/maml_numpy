#!/usr/bin/env python3
import pickle
import copy
import random
import numpy as np
import matplotlib.pyplot as plt
import argparse
from collections import defaultdict

from utils.optim import AdamOptimizer
from utils.common import GradDict # just automate some element checking (overkill)
from utils.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array, rel_error
from utils.data_generator import SinusoidGenerator

# TODO: probably add some plotting or something that shows that it actually works, rather than just the loss. Basically add a test. 
# TODO: how would I adapt this to be able to take more than one gradient step 
# TODO: refactor the inner_backward to maybe be used in the meta_backward, though I kind of don't like that it is all modularize, though I kind of do.
# TODO: add more capcity


# this will create a special dictionary that returns 0 if the element is not set, instead of error
# (it makes the code for updating gradients simpler)
GradDict = lambda: defaultdict(lambda: 0) 

normalize = lambda x: (x - x.mean()) / (x.std() + 1e-8)

def build_weights(hidden_dim=200):
    """Return weights to be used in forward pass"""
    # Initialize all weights (model params) with "Xavier Initialization" 
    # weight matrix init = uniform(-1, 1) / sqrt(layer_input)
    # bias init = zeros()
    H = hidden_dim
    d = {}
    d['W1'] = (-1 + 2*np.random.rand(1, H)) / np.sqrt(1)
    d['b1'] = np.zeros(H)
    d['W2'] = (-1 + 2*np.random.rand(H, 1)) / np.sqrt(H)
    d['b2'] = np.zeros(1)

    # Cast all parameters to the correct datatype
    for k, v in d.items():
        d[k] = v.astype(np.float32)
    return d

def save_weights(weights, filename, quiet=False):
    with open(filename, 'wb') as f:
        pickle.dump(weights, f)
    if not quiet:
        print('weights saved to {}'.format(filename))

def load_weights(filename, quiet=False):
    with open(filename, 'rb') as f:
        weights = pickle.load(f)
    if not quiet:
        print('weights loaded from {}'.format(filename))
    return weights


class Network(object):
    """BYOW: Bring Your Own Weights

    Hard-code operations for a 2 layer neural network
    """
    def __init__(self, alpha=0.01, normalized=normalize):
        self.ALPHA = alpha
        self.normalized = normalized

    def inner_forward(self, x_a, w):
        """submodule for forward pass"""
        W1, b1, W2, b2 = w['W1'], w['b1'], w['W2'], w['b2']

        affine1_a = x_a.dot(W1) + b1
        relu1_a = np.maximum(0, affine1_a)
        pred_a = relu1_a.dot(W2) + b2 

        cache = dict(x_a=x_a, affine1_a=affine1_a, relu1_a=relu1_a)
        return pred_a, cache

    def inner_backward(self, dout_a, weights, cache):
        """just for fine-tuning at the end"""
        w = weights; c = cache
        W1, b1, W2, b2 = w['W1'], w['b1'], w['W2'], w['b2']

        drelu1_a = dout_a.dot(W2.T)
        dW2 = cache['relu1_a'].T.dot(dout_a)
        db2 = np.sum(dout_a, axis=0)

        daffine1_a = np.where(cache['affine1_a'] > 0, drelu1_a, 0)

        dW1 = c['x_a'].T.dot(daffine1_a)
        db1 = np.sum(daffine1_a, axis=0)

        # grad steps
        new_weights = {}
        new_weights['W1'] = W1 - self.ALPHA*self.normalized(dW1)
        new_weights['b1'] = b1 - self.ALPHA*self.normalized(db1)
        new_weights['W2'] = W2 - self.ALPHA*self.normalized(dW2)
        new_weights['b2'] = b2 - self.ALPHA*self.normalized(db2)
        return new_weights


    def meta_forward(self, x_a, x_b, label_a, weights, cache=None):
        w = weights
        W1, b1, W2, b2 = w['W1'], w['b1'], w['W2'], w['b2']

        # standard forward and backward computations
        # (a)
        pred_a, inner_cache = self.inner_forward(x_a, w)

        dout_a = 2*(pred_a - label_a)

        drelu1_a = dout_a.dot(W2.T)
        dW2 = inner_cache['relu1_a'].T.dot(dout_a)
        db2 = np.sum(dout_a, axis=0)

        daffine1_a = np.where(inner_cache['affine1_a'] > 0, drelu1_a, 0)

        dW1 = x_a.T.dot(daffine1_a)
        db1 = np.sum(daffine1_a, axis=0)

        # Forward on fast weights
        # (b)

        # grad steps
        W1_prime = W1 - self.ALPHA*dW1
        b1_prime = b1 - self.ALPHA*db1
        W2_prime = W2 - self.ALPHA*dW2
        b2_prime = b2 - self.ALPHA*db2

        affine1_b = x_b.dot(W1_prime) + b1_prime
        relu1_b = np.maximum(0, affine1_b)
        pred_b = relu1_b.dot(W2_prime) + b2_prime

        if cache:
            outer_cache = dict(dout_a=dout_a, x_b=x_b, affine1_b=affine1_b, relu1_b=relu1_b, W2_prime=W2_prime)
            return pred_b, {**inner_cache, **outer_cache}
        else:
            return pred_b
    
    def meta_backward(self, dout_b, weights, cache, grads=None):
        c = cache; w = weights # short 
        W1, b1, W2, b2 = w['W1'], w['b1'], w['W2'], w['b2']

        # deriv w.r.t b (lower half)
        # d 1st layer
        dW2_prime = c['relu1_b'].T.dot(dout_b)
        db2_prime = np.sum(dout_b, axis=0)
        drelu1_b = dout_b.dot(c['W2_prime'].T)

        daffine1_b = np.where(c['affine1_b'] > 0, drelu1_b, 0)
        # d 2nd layer
        dW1_prime = c['x_b'].T.dot(daffine1_b)
        db1_prime = np.sum(daffine1_b, axis=0)

        # deriv w.r.t a (upper half)

        # going back through the gradient descent step
        dW1 = dW1_prime
        db1 = db1_prime
        dW2 = dW2_prime
        db2 = db2_prime

        ddW1 = dW1_prime * -self.ALPHA
        ddb1 = db1_prime * -self.ALPHA
        ddW2 = dW2_prime * -self.ALPHA
        ddb2 = db2_prime * -self.ALPHA

        # backpropping through the first backprop
        ddout_a = c['relu1_a'].dot(ddW2)
        ddout_a += ddb2
        drelu1_a = c['dout_a'].dot(ddW2.T) # shortcut back because of the grad dependency

        ddaffine1_a = c['x_a'].dot(ddW1) 
        ddaffine1_a += ddb1
        ddrelu1_a = np.where(c['affine1_a'] > 0, ddaffine1_a, 0)

        dW2 += ddrelu1_a.T.dot(c['dout_a'])

        ddout_a += ddrelu1_a.dot(W2)

        dpred_a = ddout_a * 2 # = dout_a

        dW2 += c['relu1_a'].T.dot(dpred_a)
        db2 += np.sum(dpred_a, axis=0)

        drelu1_a += dpred_a.dot(W2.T)

        daffine1_a = np.where(c['affine1_a'] > 0, drelu1_a, 0)

        dW1 += c['x_a'].T.dot(daffine1_a)
        db1 += np.sum(daffine1_a, axis=0)

        if grads is not None:
            # update gradients 
            grads['W1'] += self.normalized(dW1)
            grads['b1'] += self.normalized(db1)
            grads['W2'] += self.normalized(dW2)
            grads['b2'] += self.normalized(db2)

   
def gradcheck():
    # Test the network gradient 
    nn = Network(normalized=lambda x: x)
    grads = GradDict()

    np.random.seed(231)
    x_a = np.random.randn(15, 1)
    x_b = np.random.randn(15, 1)
    label = np.random.randn(15, 1)
    W1 = np.random.randn(1, 40)
    b1 = np.random.randn(40)
    W2 = np.random.randn(40, 1)
    b2 = np.random.randn(1)

    dout = np.random.randn(15, 1)

    weights = w = {}
    w['W1'] = W1
    w['b1'] = b1
    w['W2'] = W2
    w['b2'] = b2

    def rep_param(weights, name, val):
        clean_params = copy.deepcopy(weights)
        clean_params[name] = val
        return clean_params

    dW1_num = eval_numerical_gradient_array(lambda w: nn.meta_forward(x_a, x_b, label, rep_param(weights, 'W1', w)), W1, dout)
    db1_num = eval_numerical_gradient_array(lambda b: nn.meta_forward(x_a, x_b, label, rep_param(weights, 'b1', b)), b1, dout)
    dW2_num = eval_numerical_gradient_array(lambda w: nn.meta_forward(x_a, x_b, label, rep_param(weights, 'W2', w)), W2, dout)
    db2_num = eval_numerical_gradient_array(lambda b: nn.meta_forward(x_a, x_b, label, rep_param(weights, 'b2', b)), b2, dout)

    out, cache = nn.meta_forward(x_a, x_b, label, weights, cache=True)
    nn.meta_backward(dout, weights, cache, grads)

    # The error should be around 1e-10
    print()
    print('Testing meta-learning NN backward function:')
    print('dW1 error: ', rel_error(dW1_num, grads['W1']))
    print('db1 error: ', rel_error(db1_num, grads['b1']))
    print('dW2 error: ', rel_error(dW2_num, grads['W2']))
    print('db2 error: ', rel_error(db2_num, grads['b2']))
    print()

def test():
    """take one grad step using a minibatch of size 5 and see how well it works

    basically what they show in Figure 2 of:
    https://arxiv.org/pdf/1703.03400.pdf
    """ 
    nn = Network()
    pre_weights = load_weights(FLAGS.weight_path)
    random_weights = build_weights()

    # values for fine-tuning step
    N = 10
    sin_gen = SinusoidGenerator(5*N, 1) 
    x, y, amp, phase = map(lambda x: x[0], sin_gen.generate()) # grab all the first elems
    xs = np.split(x, N)
    ys = np.split(y, N)

    new_weights = pre_weights.copy()
    new_random_weights = random_weights.copy()
    for i in range(len(xs)):
        x = xs[i]
        y = ys[i]
        grads = GradDict()
        pred, cache = nn.inner_forward(x, new_weights)
        loss = (pred - y)**2
        dout = 2*(pred - y)
        new_weights = nn.inner_backward(dout, new_weights, cache)

    for i in range(len(xs)):
        x = xs[i]
        y = ys[i]
        grads = GradDict()
        pred, cache = nn.inner_forward(x, new_random_weights)
        loss = (pred - y)**2
        dout = 2*(pred - y)
        new_random_weights = nn.inner_backward(dout, new_random_weights, cache)


    sine_true = lambda x: amp*np.sin(x - phase)
    sine_nn = lambda x: nn.inner_forward(x, new_weights)[0]
    sine_pre = lambda x: nn.inner_forward(x, pre_weights)[0]
    sine_random = lambda x: nn.inner_forward(x, random_weights)[0]
    sine_new_random = lambda x: nn.inner_forward(x, new_random_weights)[0]

    x_vals = np.linspace(-5, 5)

    y_true = np.apply_along_axis(sine_true, 0, x_vals)
    y_nn = np.array([sine_nn(np.array(x)) for x in x_vals]).squeeze()
    y_pre = np.array([sine_pre(np.array(x)) for x in x_vals]).squeeze()
    y_random = np.array([sine_random(np.array(x)) for x in x_vals]).squeeze()
    y_new_random = np.array([sine_new_random(np.array(x)) for x in x_vals]).squeeze()

    plt.plot(x_vals, y_true, 'k', label='{:.2f}sin(x - {:.2f})'.format(amp, phase))
    plt.plot(x_vals, y_pre, 'r--', label='pre-update')
    plt.plot(x_vals, y_nn, 'r-', label='post-update')
    plt.plot(x_vals, y_random, 'g--', label='random')
    plt.plot(x_vals, y_new_random, 'g-', label='new_random')
    plt.legend()
    plt.show()


def main():
    nn = Network()
    weights = build_weights()
    optimizer = AdamOptimizer(weights, learning_rate=FLAGS.learning_rate)

    sin_gen = SinusoidGenerator(10, 25)  # update_batch * 2, meta batch size


    lr = lambda x: x * FLAGS.learning_rate

    nitr = 1e4
    for itr in range(int(nitr)):
        frac = 1.0 - (itr / nitr)

        # create a minibatch of size 25, with 10 points
        batch_x, batch_y, amp, phase = sin_gen.generate()

        inputa = batch_x[:, :5, :]
        labela = batch_y[:, :5, :]
        inputb = batch_x[:, 5:, :] # b used for testing
        labelb = batch_y[:, 5:, :]
        
        # META BATCH
        grads = GradDict() # zero grads
        losses = []
        for batch_i in range(len(inputa)):
            ia, la, ib, lb = inputa[batch_i], labela[batch_i], inputb[batch_i], labelb[batch_i]
            pred_b, cache = nn.meta_forward(ia, ib, la, weights, cache=True)
            losses.append((pred_b - lb)**2)
            dout_b = 2*(pred_b - lb)
            nn.meta_backward(dout_b, weights, cache, grads)
        optimizer.apply_gradients(weights, grads, learning_rate=lr(frac))
        if itr % 100 == 0:
            print("[itr: {}] Loss = {}".format(itr, np.sum(losses)))

    save_weights(weights, FLAGS.weight_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MAML')
    parser.add_argument('--gradcheck', type=int, default=0, help='Run gradient check and other tests')
    parser.add_argument('--test', type=int, default=0, help='Run test on trained network to see if it works')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_path', type=str, default='trained_maml_weights.pkl', help='File name to save and load weights')
    FLAGS = parser.parse_args()
    
    if FLAGS.gradcheck:
        gradcheck()
        exit(0)

    if FLAGS.test:
        test()
        exit(0)

    main()



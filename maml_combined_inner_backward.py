#!/usr/bin/env python3
import os
import pickle
import copy
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl; 
mpl.rcParams["savefig.directory"] = '~/Desktop'#$os.chdir(os.path.dirname(__file__))
import argparse
from collections import defaultdict

from utils.optim import AdamOptimizer
from utils.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array, rel_error
from utils.data_generator import SinusoidGenerator

"""
This file contains logic for training a fully-connected neural network with
2 hidden layers using the Model-Agnostic Meta-Learning (MAML) algorithm.

It is designed to solve the toy sinusoid meta-learning problem presented in the MAML paper, 
and uses the same architecture as presented in the paper.

Passing the `--gradcheck=1` flag, will run finite differences gradient check
on the meta forward and backward to ensure correct implementation.

After training a network, you can pass the `--test=1` flag to compare against
a joint-trained and random network baseline.
"""

# TODO: refactor the inner_backward to maybe be used in the meta_backward, though I kind of don't like that it is all modularize, though I kind of do.  this could be added right 
# after the inner_forward in the meta_forward, and it would have to cache stuff
# TODO: what is FOMAML for me?

# special dictionary to return 0 if element does not exist (makes gradient code simpler)
GradDict = lambda: defaultdict(lambda: 0) 
normalize = lambda x: (x - x.mean()) / (x.std() + 1e-8)

def build_weights(hidden_dims=(64, 64)):
    """Return weights to be used in forward pass"""
    # Initialize all weights (model params) with "Xavier Initialization" 
    # weight matrix init = uniform(-1, 1) / sqrt(layer_input)
    # bias init = zeros()
    H1, H2 = hidden_dims
    d = {}
    d['W1'] = (-1 + 2*np.random.rand(1, H1)) / np.sqrt(1)
    d['b1'] = np.zeros(H1)
    d['W2'] = (-1 + 2*np.random.rand(H1, H2)) / np.sqrt(H1)
    d['b2'] = np.zeros(H2)
    d['W3'] = (-1 + 2*np.random.rand(H2, 1)) / np.sqrt(H2)
    d['b3'] = np.zeros(1)

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
    """
    Hard-code operations for a 3 layer neural network
    """
    def __init__(self, inner_lr=0.01, normalize=normalize):
        self.inner_lr = inner_lr  # alpha in the paper
        self.normalize = normalize  # function to normalize gradients before applying them to weights (helps with stability)

    def inner_forward(self, x_a, weights):
        """Submodule for forward pass. This is what standard forward pass of network looks like"""
        w = weights
        W1, b1, W2, b2, W3, b3 = w['W1'], w['b1'], w['W2'], w['b2'], w['W3'], w['b3']

        # layer 1
        affine1_a = x_a.dot(W1) + b1
        relu1_a = np.maximum(0, affine1_a)
        # layer 2
        affine2_a = relu1_a.dot(W2) + b2 
        relu2_a = np.maximum(0, affine2_a)
        # layer 3
        pred_a = relu2_a.dot(W3) + b3

        cache = dict(x_a=x_a, affine1_a=affine1_a, relu1_a=relu1_a, affine2_a=affine2_a, relu2_a=relu2_a)
        return pred_a, cache

    def inner_backward(self, dout_a, weights, cache, grads=None, normalize=None):
        """just for fine-tuning at the end"""
        normalize = normalize or self.normalize
        w = weights; c = cache; d = grads

        W1, b1, W2, b2, W3, b3 = w['W1'], w['b1'], w['W2'], w['b2'], w['W3'], w['b3']

        drelu2_a = dout_a.dot(W3.T)
        dW3 = c['relu2_a'].T.dot(dout_a)
        db3 = np.sum(dout_a, axis=0)

        daffine2_a = np.where(c['affine2_a'] > 0, drelu2_a, 0)

        drelu1_a = daffine2_a.dot(W2.T)
        dW2 = c['relu1_a'].T.dot(dout_a)
        db2 = np.sum(dout_a, axis=0)

        daffine1_a = np.where(c['affine1_a'] > 0, drelu1_a, 0)

        dW1 = c['x_a'].T.dot(daffine1_a)
        db1 = np.sum(daffine1_a, axis=0)

        # TODO: merge

        d['relu2_a'] += d['pred_a'].dot(d['W3'].T)


        drelu2_a += dpred_a.dot(W3.T)
        db3 += np.sum(dpred_a, axis=0)
        dW3 += c['relu2_a'].T.dot(dpred_a)

        daffine2_a = np.where(c['affine2_a'] > 0, drelu2_a, 0)

        drelu1_a += daffine2_a.dot(W2.T)
        dW2 += c['relu1_a'].T.dot(daffine2_a)
        db2 += np.sum(daffine2_a, axis=0)

        daffine1_a = np.where(c['affine1_a'] > 0, drelu1_a, 0)

        dW1 += c['x_a'].T.dot(daffine1_a)
        db1 += np.sum(daffine1_a, axis=0)

        # grad steps
        if grads is None:
            new_weights = {}
            new_weights['W1'] = W1 - self.inner_lr*normalize(dW1)
            new_weights['b1'] = b1 - self.inner_lr*normalize(db1)
            new_weights['W2'] = W2 - self.inner_lr*normalize(dW2)
            new_weights['b2'] = b2 - self.inner_lr*normalize(db2)
            new_weights['W3'] = W3 - self.inner_lr*normalize(dW3)
            new_weights['b3'] = b3 - self.inner_lr*normalize(db3)
            return new_weights
        else:
            grads['W1'] += normalize(dW1)
            grads['b1'] += normalize(db1)
            grads['W2'] += normalize(dW2)
            grads['b2'] += normalize(db2)
            grads['W3'] += normalize(dW3)
            grads['b3'] += normalize(db3)
            return grads


    def meta_forward(self, x_a, x_b, label_a, weights, cache=None):
        w = weights
        W1, b1, W2, b2, W3, b3 = w['W1'], w['b1'], w['W2'], w['b2'], w['W3'], w['b3']

        # standard forward and backward computations
        # (a)
        pred_a, inner_cache = self.inner_forward(x_a, w)

        dout_a = 2*(pred_a - label_a)

        dW3 = inner_cache['relu2_a'].T.dot(dout_a)
        db3 = np.sum(dout_a, axis=0)
        drelu2_a = dout_a.dot(W3.T)

        daffine2_a = np.where(inner_cache['affine2_a'] > 0, drelu2_a, 0)

        dW2 = inner_cache['relu1_a'].T.dot(daffine2_a)
        db2 = np.sum(daffine2_a, axis=0)
        drelu1_a = daffine2_a.dot(W2.T)

        daffine1_a = np.where(inner_cache['affine1_a'] > 0, drelu1_a, 0)

        dW1 = x_a.T.dot(daffine1_a)
        db1 = np.sum(daffine1_a, axis=0)

        # Forward on fast weights
        # (b)

        # grad steps
        W1_prime = W1 - self.inner_lr*dW1
        b1_prime = b1 - self.inner_lr*db1
        W2_prime = W2 - self.inner_lr*dW2
        b2_prime = b2 - self.inner_lr*db2
        W3_prime = W3 - self.inner_lr*dW3
        b3_prime = b3 - self.inner_lr*db3

        affine1_b = x_b.dot(W1_prime) + b1_prime
        relu1_b = np.maximum(0, affine1_b)
        affine2_b = relu1_b.dot(W2_prime) + b2_prime
        relu2_b = np.maximum(0, affine2_b)
        pred_b = relu2_b.dot(W3_prime) + b3_prime

        if cache:
            outer_cache = dict(dout_a=dout_a, x_b=x_b, affine1_b=affine1_b, relu1_b=relu1_b, affine2_b=affine2_b, relu2_b=relu2_b, daffine2_a=daffine2_a, W2_prime=W2_prime, W3_prime=W3_prime)
            return pred_b, {**inner_cache, **outer_cache}
        else:
            return pred_b
    
    def meta_backward(self, dout_b, weights, cache, grads=None):
        c = cache; w = weights # short 
        W1, b1, W2, b2, W3, b3 = w['W1'], w['b1'], w['W2'], w['b2'], w['W3'], w['b3']

        # deriv w.r.t b (lower half)
        # d 3rd layer
        drelu2_b = dout_b.dot(c['W3_prime'].T)
        dW3_prime = c['relu2_b'].T.dot(dout_b)
        db3_prime = np.sum(dout_b, axis=0)

        daffine2_b = np.where(c['affine2_b'] > 0, drelu2_b, 0)

        # d 2rd layer
        drelu1_b = daffine2_b.dot(c['W2_prime'].T)
        dW2_prime = c['relu1_b'].T.dot(daffine2_b)
        db2_prime = np.sum(daffine2_b, axis=0)

        daffine1_b = np.where(c['affine1_b'] > 0, drelu1_b, 0)

        # d 1st layer
        dW1_prime = c['x_b'].T.dot(daffine1_b)
        db1_prime = np.sum(daffine1_b, axis=0)

        # deriv w.r.t a (upper half)
        # going back through the gradient descent step
        dW1 = dW1_prime
        db1 = db1_prime
        dW2 = dW2_prime
        db2 = db2_prime
        dW3 = dW3_prime
        db3 = db3_prime

        ddW1 = dW1_prime * -self.inner_lr
        ddb1 = db1_prime * -self.inner_lr
        ddW2 = dW2_prime * -self.inner_lr
        ddb2 = db2_prime * -self.inner_lr
        ddW3 = dW3_prime * -self.inner_lr
        ddb3 = db3_prime * -self.inner_lr

        # backpropping through the first backprop

        # start with dW1's
        #dx = c['daffine1_a'].dot(ddW1.T) # don't need it unless we backprop through input (x)
        ddaffine1_a = c['x_a'].dot(ddW1) 
        ddaffine1_a += ddb1

        ddrelu1_a = np.where(c['affine1_a'] > 0, ddaffine1_a, 0)

        ddaffine2_a = ddrelu1_a.dot(W2)
        dW2 += ddrelu1_a.T.dot(c['daffine2_a'])

        # dW2's
        drelu1_a = c['daffine2_a'].dot(ddW2.T) # shortcut back because of the grad dependency
        ddaffine2_a += ddb2
        ddaffine2_a += c['relu1_a'].dot(ddW2)

        ddrelu2_a = np.where(c['affine2_a'] > 0, ddaffine2_a, 0)

        ddout_a = ddrelu2_a.dot(W3)
        dW3 += ddrelu2_a.T.dot(c['dout_a'])

        # dW3's
        drelu2_a = c['dout_a'].dot(ddW3.T) # shortcut back because of the grad dependency
        ddout_a += ddb3
        ddout_a += c['relu2_a'].dot(ddW3)

        # back through the first forward
        dpred_a = ddout_a * 2 

        # Combine grads from first inner_forward because we already have a function to compute them
        first_inner_grads = self.inner_backward(dpred_a, weights, cache, grads=grads, normalize=lambda x: x)  # don't normalize here because we will normalize them anyway

        if grads is not None:
            # update gradients 
            grads['W1'] += self.normalize(dW1 + first_inner_grads['W1'])
            grads['b1'] += self.normalize(db1 + first_inner_grads['b1'])
            grads['W2'] += self.normalize(dW2 + first_inner_grads['W2'])
            grads['b2'] += self.normalize(db2 + first_inner_grads['b2'])
            grads['W3'] += self.normalize(dW3 + first_inner_grads['W3'])
            grads['b3'] += self.normalize(db3 + first_inner_grads['b3'])

   
def gradcheck():
    # Test the network gradient 
    nn = Network(normalize=lambda x: x) # don't normalize gradients so we can check validity 
    grads = GradDict()

    np.random.seed(231)
    x_a = np.random.randn(15, 1)
    x_b = np.random.randn(15, 1)
    label = np.random.randn(15, 1)
    W1 = np.random.randn(1, 40)
    b1 = np.random.randn(40)
    W2 = np.random.randn(40, 40)
    b2 = np.random.randn(40)
    W3 = np.random.randn(40, 1)
    b3 = np.random.randn(1)

    dout = np.random.randn(15, 1)

    weights = w = {}
    w['W1'] = W1
    w['b1'] = b1
    w['W2'] = W2
    w['b2'] = b2
    w['W3'] = W3
    w['b3'] = b3

    def rep_param(weights, name, val):
        clean_params = copy.deepcopy(weights)
        clean_params[name] = val
        return clean_params

    dW1_num = eval_numerical_gradient_array(lambda w: nn.meta_forward(x_a, x_b, label, rep_param(weights, 'W1', w)), W1, dout, h=1e-5)
    db1_num = eval_numerical_gradient_array(lambda b: nn.meta_forward(x_a, x_b, label, rep_param(weights, 'b1', b)), b1, dout, h=1e-5)
    dW2_num = eval_numerical_gradient_array(lambda w: nn.meta_forward(x_a, x_b, label, rep_param(weights, 'W2', w)), W2, dout, h=1e-5)
    db2_num = eval_numerical_gradient_array(lambda b: nn.meta_forward(x_a, x_b, label, rep_param(weights, 'b2', b)), b2, dout, h=1e-5)
    dW3_num = eval_numerical_gradient_array(lambda w: nn.meta_forward(x_a, x_b, label, rep_param(weights, 'W3', w)), W3, dout, h=1e-5)
    db3_num = eval_numerical_gradient_array(lambda b: nn.meta_forward(x_a, x_b, label, rep_param(weights, 'b3', b)), b3, dout, h=1e-5)

    out, cache = nn.meta_forward(x_a, x_b, label, weights, cache=True)
    nn.meta_backward(dout, weights, cache, grads)

    # The error should be around 1e-10
    print()
    print('Testing meta-learning NN backward function:')
    print('dW1 error: ', rel_error(dW1_num, grads['W1']))
    print('db1 error: ', rel_error(db1_num, grads['b1']))
    print('dW2 error: ', rel_error(dW2_num, grads['W2']))
    print('db2 error: ', rel_error(db2_num, grads['b2']))
    print('dW3 error: ', rel_error(dW3_num, grads['W3']))
    print('db3 error: ', rel_error(db3_num, grads['b3']))
    print()

def test():
    """take one grad step using a minibatch of size 5 and see how well it works

    basically what they show in Figure 2 of:
    https://arxiv.org/pdf/1703.03400.pdf
    """ 
    nn = Network(inner_lr=FLAGS.inner_lr)
    pre_weights = load_weights(FLAGS.weight_path)
    baseline_weights = load_weights('baseline_'+FLAGS.weight_path)
    random_weights = build_weights()

    # values for fine-tuning step
    N = 10
    sin_gen = SinusoidGenerator(5*N, 1) 
    x, y, amp, phase = map(lambda x: x[0], sin_gen.generate()) # grab all the first elems
    xs = np.split(x, N)
    ys = np.split(y, N)

    new_weights = pre_weights.copy()
    new_random_weights = random_weights.copy()
    new_baseline_weights = baseline_weights.copy()
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

    for i in range(len(xs)):
        x = xs[i]
        y = ys[i]
        grads = GradDict()
        pred, cache = nn.inner_forward(x, new_baseline_weights)
        loss = (pred - y)**2
        dout = 2*(pred - y)
        new_baseline_weights = nn.inner_backward(dout, new_baseline_weights, cache)


    sine_true = lambda x: amp*np.sin(x - phase)
    sine_pre = lambda x: nn.inner_forward(x, pre_weights)[0]
    sine_nn = lambda x: nn.inner_forward(x, new_weights)[0]
    sine_pre_baseline = lambda x: nn.inner_forward(x, baseline_weights)[0]
    sine_baseline = lambda x: nn.inner_forward(x, new_baseline_weights)[0]
    sine_random = lambda x: nn.inner_forward(x, random_weights)[0]
    sine_new_random = lambda x: nn.inner_forward(x, new_random_weights)[0]

    x_vals = np.linspace(-5, 5)

    y_true = np.apply_along_axis(sine_true, 0, x_vals)

    y_pre = np.array([sine_pre(np.array(x)) for x in x_vals]).squeeze()
    y_nn = np.array([sine_nn(np.array(x)) for x in x_vals]).squeeze()

    y_pre_baseline = np.array([sine_pre_baseline(np.array(x)) for x in x_vals]).squeeze()
    y_baseline = np.array([sine_baseline(np.array(x)) for x in x_vals]).squeeze()

    y_random = np.array([sine_random(np.array(x)) for x in x_vals]).squeeze()
    y_new_random = np.array([sine_new_random(np.array(x)) for x in x_vals]).squeeze()

    plt.plot(x_vals, y_true, 'k', label='{:.2f}sin(x - {:.2f})'.format(amp, phase))
    plt.plot(x_vals, y_pre, 'r--', label='pre-update')
    plt.plot(x_vals, y_nn, 'r-', label='post-update')

    plt.plot(x_vals, y_pre_baseline, 'g--', label='baseline')
    plt.plot(x_vals, y_baseline, 'g-', label='new_baseline')

    #plt.plot(x_vals, y_random, 'g--', label='random')
    #plt.plot(x_vals, y_new_random, 'g-', label='new_random')
    plt.legend()
    plt.title("MAML sinusoid matching after {} fine-tuning update".format(N))
    plt.show()


def train():
    nn = Network(inner_lr=FLAGS.inner_lr)
    weights = build_weights()
    baseline_weights = build_weights()
    optimizer = AdamOptimizer(weights, learning_rate=FLAGS.meta_lr)
    baseline_optimizer = AdamOptimizer(baseline_weights, learning_rate=FLAGS.meta_lr)

    sin_gen = SinusoidGenerator(10, 25)  # update_batch * 2, meta batch size

    nitr = 1e4
    for itr in range(int(nitr)):

        # create a minibatch of size 25, with 10 points
        batch_x, batch_y, amp, phase = sin_gen.generate()

        inputa = batch_x[:, :5, :]
        labela = batch_y[:, :5, :]
        inputb = batch_x[:, 5:, :] # b used for testing
        labelb = batch_y[:, 5:, :]
        
        # META BATCH
        grads = GradDict() # zero grads
        baseline_grads = GradDict() # zero grads
        losses = []
        for batch_i in range(len(inputa)):
            ia, la, ib, lb = inputa[batch_i], labela[batch_i], inputb[batch_i], labelb[batch_i]

            pred_b, cache = nn.meta_forward(ia, ib, la, weights, cache=True)
            losses.append((pred_b - lb)**2)
            dout_b = 2*(pred_b - lb)
            nn.meta_backward(dout_b, weights, cache, grads)


            baseline_pred_b, baseline_cache = nn.meta_forward(ia, ib, la, baseline_weights, cache=True)
            #losses.append((pred_b - lb)**2)
            dout_b = 2*(pred_b - lb)
            nn.meta_backward(dout_b, baseline_weights, baseline_cache, baseline_grads)


        optimizer.apply_gradients(weights, grads, learning_rate=FLAGS.meta_lr)
        if itr % 100 == 0:
            print("[itr: {}] Loss = {}".format(itr, np.sum(losses)))

    save_weights(weights, FLAGS.weight_path)
    save_weights(baseline_weights, "baseline_"+FLAGS.weight_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MAML')
    parser.add_argument('--gradcheck', type=int, default=0, help='Run gradient check and other tests')
    parser.add_argument('--test', type=int, default=0, help='Run test on trained network to see if it works')
    parser.add_argument('--meta_lr', type=float, default=1e-3, help='Meta learning rate')
    parser.add_argument('--inner_lr', type=float, default=1e-3, help='Inner learning rate')
    parser.add_argument('--weight_path', type=str, default='trained_maml_weights.pkl', help='File name to save and load weights')
    FLAGS = parser.parse_args()
    
    if FLAGS.gradcheck:
        gradcheck()
        exit(0)

    if FLAGS.test:
        test()
        exit(0)

    train()



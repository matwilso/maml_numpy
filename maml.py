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


# special dictionary to return 0 if element does not exist (makes gradient code simpler)
GradDict = lambda: defaultdict(lambda: 0) 
normalize = lambda x: (x - x.mean()) / (x.std() + 1e-8)

# weight util functions
def build_weights(hidden_dims=(64, 64)):
    """Return dictionary on neural network weights"""
    # Initialize all weights (model params) with "Xavier Initialization" 
    # weight matrix init = uniform(-1, 1) / sqrt(layer_input)
    # bias init = zeros()
    H1, H2 = hidden_dims
    w = {}
    w['W1'] = (-1 + 2*np.random.rand(1, H1)) / np.sqrt(1)
    w['b1'] = np.zeros(H1)
    w['W2'] = (-1 + 2*np.random.rand(H1, H2)) / np.sqrt(H1)
    w['b2'] = np.zeros(H2)
    w['W3'] = (-1 + 2*np.random.rand(H2, 1)) / np.sqrt(H2)
    w['b3'] = np.zeros(1)

    # Cast all parameters to the correct datatype
    for k, v in w.items():
        w[k] = v.astype(np.float32)
    return w

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
    Forward and backward pass logic for 3 layer neural network
    (see https://github.com/matwilso/maml_numpy#derivation for derivation)
    """

    def __init__(self, inner_lr=0.01, normalize=normalize):
        self.inner_lr = inner_lr  # alpha in the paper
        self.normalize = normalize  # function to normalize gradients before applying them to weights (helps with stability)

    def inner_forward(self, x_a, weights, cache={}):
        """Submodule for meta_forward. This is just a standard forward pass for a neural net.

        Args:
            x_a (ndarray): Example or examples of sinusoid from given phase, amplitude.  
            weights (dict): Dictionary of weights and biases for neural net
            cache (dict): Pass in dictionary to be updated with values needed in meta_backward

        Returns:
            pred_a (ndarray): Predicted values for example(s) x_a
        """
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

        cache.update(dict(x_a=x_a, affine1_a=affine1_a, relu1_a=relu1_a, affine2_a=affine2_a, relu2_a=relu2_a))
        return pred_a

    def inner_backward(self, dout_a, weights, cache, grads=GradDict(), lr=None):
        """For fine-tuning network at meta-test time

        (Although this has some repeated code from meta_backward, it was hard to 
        use as a subprocess for meta_backward.  It required several changes in 
        code and made things more confusing.)

        Args:
            dout_a (ndarray): Gradient of output (usually loss)
            weights (dict): Dictionary of weights and biases for neural net
            cache (dict): Dictionary of relevant values from forward pass

        Returns:
            dict: New dictionary, with updated weights
        """
        w = weights; c = cache
        W1, b1, W2, b2, W3, b3 = w['W1'], w['b1'], w['W2'], w['b2'], w['W3'], w['b3']
        lr = lr or self.inner_lr

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

        grads['W1'] += dW1
        grads['b1'] += db1
        grads['W2'] += dW2
        grads['b2'] += db2
        grads['W3'] += dW3
        grads['b3'] += db3

        # Return new weights (for fine-tuning)
        new_weights = {}
        new_weights['W1'] = W1 - lr*self.normalize(dW1)
        new_weights['b1'] = b1 - lr*self.normalize(db1)
        new_weights['W2'] = W2 - lr*self.normalize(dW2)
        new_weights['b2'] = b2 - lr*self.normalize(db2)
        new_weights['W3'] = W3 - lr*self.normalize(dW3)
        new_weights['b3'] = b3 - lr*self.normalize(db3)
        return new_weights


    def meta_forward(self, x_a, x_b, label_a, weights, cache={}):
        """Full forward pass for MAML. Does a inner_forward, backprop, and gradient 
        update.  This will all be backpropped through w.r.t. weights in meta_backward

        Args:
            x_a (ndarray): Example or examples of sinusoid from given phase, amplitude.  
            x_b (ndarray): Independent example(s) from same phase, amplitude as x_a's
            label_a (ndarray): Ground truth labels for x_a
            weights (dict): Dictionary of weights and biases for neural net
            cache (dict): Pass in dictionary to be updated with values needed in meta_backward

        Returns:
            pred_b (ndarray): Predicted values for example(s) x_b
        """
        w = weights
        W1, b1, W2, b2, W3, b3 = w['W1'], w['b1'], w['W2'], w['b2'], w['W3'], w['b3']

        # A: inner
        # standard forward and backward computations
        inner_cache = {}
        pred_a = self.inner_forward(x_a, w, inner_cache)

        # inner loss
        dout_a = 2*(pred_a - label_a) 

        # d 3rd layer
        dW3 = inner_cache['relu2_a'].T.dot(dout_a)
        db3 = np.sum(dout_a, axis=0)
        drelu2_a = dout_a.dot(W3.T)

        daffine2_a = np.where(inner_cache['affine2_a'] > 0, drelu2_a, 0)

        # d 2nd layer
        dW2 = inner_cache['relu1_a'].T.dot(daffine2_a)
        db2 = np.sum(daffine2_a, axis=0)
        drelu1_a = daffine2_a.dot(W2.T)

        daffine1_a = np.where(inner_cache['affine1_a'] > 0, drelu1_a, 0)

        # d 1st layer
        dW1 = x_a.T.dot(daffine1_a)
        db1 = np.sum(daffine1_a, axis=0)

        # Forward on fast weights
        # B: meta/outer
        # SGD step is baked into forward pass, representing optimizing through fine-tuning
        # Theta prime in the paper. Also called fast_weights in Finn's TF implementation
        W1_prime = W1 - self.inner_lr*dW1
        b1_prime = b1 - self.inner_lr*db1
        W2_prime = W2 - self.inner_lr*dW2
        b2_prime = b2 - self.inner_lr*db2
        W3_prime = W3 - self.inner_lr*dW3
        b3_prime = b3 - self.inner_lr*db3

        # Do another forward pass with the fast weights, to predict B example
        affine1_b = x_b.dot(W1_prime) + b1_prime
        relu1_b = np.maximum(0, affine1_b)
        affine2_b = relu1_b.dot(W2_prime) + b2_prime
        relu2_b = np.maximum(0, affine2_b)
        pred_b = relu2_b.dot(W3_prime) + b3_prime

        # Cache relevant values for meta backpropping
        outer_cache = dict(dout_a=dout_a, x_b=x_b, affine1_b=affine1_b, relu1_b=relu1_b, affine2_b=affine2_b, relu2_b=relu2_b, daffine2_a=daffine2_a, W2_prime=W2_prime, W3_prime=W3_prime)
        cache.update(inner_cache)
        cache.update(outer_cache)

        return pred_b
    
    def meta_backward(self, dout_b, weights, cache, grads=GradDict()):
        """Full backward pass for MAML. Through all operations from forward pass

        Args:
            dout_b (ndarray): Gradient signal of network output (usually loss gradient)
            weights (dict): Dictionary of weights and biases used in forward pass
            cache (dict): Dictionary of relevant values from forward pass
            grads (dict): Pass in dictionary to be updated with weight gradients
        """
        c = cache; w = weights 
        W1, b1, W2, b2, W3, b3 = w['W1'], w['b1'], w['W2'], w['b2'], w['W3'], w['b3']

        # First, backprop through the B network pass
        # d 3rd layer
        drelu2_b = dout_b.dot(c['W3_prime'].T)
        dW3_prime = c['relu2_b'].T.dot(dout_b)
        db3_prime = np.sum(dout_b, axis=0)

        daffine2_b = np.where(c['affine2_b'] > 0, drelu2_b, 0)

        # d 2nd layer
        drelu1_b = daffine2_b.dot(c['W2_prime'].T)
        dW2_prime = c['relu1_b'].T.dot(daffine2_b)
        db2_prime = np.sum(daffine2_b, axis=0)

        daffine1_b = np.where(c['affine1_b'] > 0, drelu1_b, 0)

        # d 1st layer
        dW1_prime = c['x_b'].T.dot(daffine1_b)
        db1_prime = np.sum(daffine1_b, axis=0)

        # Next, backprop through the gradient descent step
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

        # Then, backprop through the first backprop
        # start with dW1's
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

        # Finally, backprop through the first forward
        dpred_a = ddout_a * 2 

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

        # update gradients 
        grads['W1'] += self.normalize(dW1)
        grads['b1'] += self.normalize(db1)
        grads['W2'] += self.normalize(dW2)
        grads['b2'] += self.normalize(db2)
        grads['W3'] += self.normalize(dW3)
        grads['b3'] += self.normalize(db3)

   
def gradcheck():
    # Test the network gradient 
    nn = Network(normalize=lambda x: x) # don't normalize gradients so we can check validity 
    grads = GradDict()  # initialize grads to 0
    # dummy inputs, labels, and fake backwards gradient signal
    x_a = np.random.randn(15, 1)
    x_b = np.random.randn(15, 1)
    label = np.random.randn(15, 1)
    dout = np.random.randn(15, 1)
    # make weights. don't use build_weights here because this is more stable
    W1 = np.random.randn(1, 40)
    b1 = np.random.randn(40)
    W2 = np.random.randn(40, 40)
    b2 = np.random.randn(40)
    W3 = np.random.randn(40, 1)
    b3 = np.random.randn(1)
    weights = dict(W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)

    # helper function to only change a single key of interest for independent finite differences
    def rep_param(weights, name, val):
        clean_params = copy.deepcopy(weights)
        clean_params[name] = val
        return clean_params

    # Evaluate gradients numerically, using finite differences
    numerical_grads = {}
    for key in weights:
        num_grad = eval_numerical_gradient_array(lambda w: nn.meta_forward(x_a, x_b, label, rep_param(weights, key, w)), weights[key], dout, h=1e-5)
        numerical_grads[key] = num_grad

    # Compute neural network gradients
    cache = {}
    out = nn.meta_forward(x_a, x_b, label, weights, cache=cache)
    nn.meta_backward(dout, weights, cache, grads)

    # The error should be around 1e-10
    print()
    for key in weights:
        print('d{} error: {}'.format(key, rel_error(numerical_grads[key], grads[key])))
    print()

def test():
    """Take one grad step using a minibatch of size 5 and see how well it works

    Basically what they show in Figure 2 of the paper
    """ 
    nn = Network(inner_lr=FLAGS.inner_lr)

    pre_weights = {}
    pre_weights['maml'] = load_weights(FLAGS.weight_path)
    if FLAGS.use_baseline:
        pre_weights['baseline'] = load_weights('baseline_'+FLAGS.weight_path)
    pre_weights['random'] = build_weights()

    # Generate N batches of data, with same shape as training, but that all have the same amplitude and phase
    N = 2
    #sinegen = SinusoidGenerator(FLAGS.inner_bs*N, 1, config={'input_range':[1.0,5.0]}) 
    sinegen = SinusoidGenerator(FLAGS.inner_bs*N, 1)
    x, y, amp, phase = map(lambda x: x[0], sinegen.generate()) # grab all the first elems
    xs = np.split(x, N)
    ys = np.split(y, N)

    # Copy pre-update weights for later comparison
    deepcopy = lambda weights: {key: weights[key].copy() for key in weights}
    post_weights = {}
    for key in pre_weights:
        post_weights[key] = deepcopy(pre_weights[key])

    T = 10
    # Run fine-tuning 
    for key in post_weights:
        for t in range(T):
            for i in range(len(xs)):
                x = xs[i]
                y = ys[i]
                grads = GradDict()
                cache = {}
                pred = nn.inner_forward(x, post_weights[key], cache)
                loss = (pred - y)**2
                dout = 2*(pred - y)
                post_weights[key] = nn.inner_backward(dout, post_weights[key], cache)


    colors = {'maml': 'r', 'baseline': 'b', 'random': 'g'}
    name = {'maml': 'MAML', 'baseline': 'joint training', 'random': 'random initialization'}

    sine_ground = lambda x: amp*np.sin(x - phase)
    sine_pre_pred = lambda x, key: nn.inner_forward(x, pre_weights[key])[0]
    sine_post_pred = lambda x, key: nn.inner_forward(x, post_weights[key])[0]

    x_vals = np.linspace(-5, 5)
    y_ground = np.apply_along_axis(sine_ground, 0, x_vals)


    for key in post_weights:
        y_pre = np.array([sine_pre_pred(np.array(x), key) for x in x_vals]).squeeze()
        y_nn = np.array([sine_post_pred(np.array(x), key) for x in x_vals]).squeeze()
        plt.plot(x_vals, y_ground, 'k', label='{:.2f}sin(x - {:.2f})'.format(amp, phase))
        plt.plot(np.concatenate(xs), np.concatenate(ys), 'ok', label='samples')
        plt.plot(x_vals, y_pre, colors[key]+'--', label='pre-update')
        plt.plot(x_vals, y_nn, colors[key]+'-', label='post-update')

        plt.legend()
        plt.title('Fine-tuning performance {}'.format(name[key]))
        plt.savefig(key+'.png')
        plt.show()

def train():
    nn = Network(inner_lr=FLAGS.inner_lr)
    weights = build_weights()
    optimizer = AdamOptimizer(weights, learning_rate=FLAGS.meta_lr)
    if FLAGS.use_baseline:
        baseline_weights = build_weights()
        baseline_optimizer = AdamOptimizer(baseline_weights, learning_rate=FLAGS.meta_lr)

    sinegen = SinusoidGenerator(2*FLAGS.inner_bs, 25)  # update_batch * 2, meta batch size

    try:
        nitr = int(FLAGS.num_iter)
        for itr in range(int(nitr)):
            # create a minibatch of size 25, with 10 points
            batch_x, batch_y, amp, phase = sinegen.generate()

            inputa = batch_x[:, :FLAGS.inner_bs :]
            labela = batch_y[:, :FLAGS.inner_bs :]
            inputb = batch_x[:, FLAGS.inner_bs :] # b used for testing
            labelb = batch_y[:, FLAGS.inner_bs :]
            
            # META BATCH
            grads = GradDict() # zero grads
            baseline_grads = GradDict() # zero grads
            losses = []
            baseline_losses = []
            for batch_i in range(len(inputa)):
                ia, la, ib, lb = inputa[batch_i], labela[batch_i], inputb[batch_i], labelb[batch_i]
                cache = {}
                pred_b = nn.meta_forward(ia, ib, la, weights, cache=cache)
                losses.append((pred_b - lb)**2)
                dout_b = 2*(pred_b - lb)
                nn.meta_backward(dout_b, weights, cache, grads)


                if FLAGS.use_baseline:
                    baseline_cache = {}
                    baseline_i = np.concatenate([ia,ib])
                    baseline_l = np.concatenate([la,lb])
                    baseline_pred = nn.inner_forward(baseline_i, baseline_weights, cache=baseline_cache)
                    baseline_losses.append((baseline_pred - baseline_l)**2)
                    dout_b = 2*(baseline_pred - baseline_l)
                    nn.inner_backward(dout_b, baseline_weights, baseline_cache, baseline_grads)

            optimizer.apply_gradients(weights, grads, learning_rate=FLAGS.meta_lr)
            if FLAGS.use_baseline:
                baseline_optimizer.apply_gradients(baseline_weights, baseline_grads, learning_rate=FLAGS.meta_lr)
            if itr % 100 == 0:
                if FLAGS.use_baseline:
                    print("[itr: {}] MAML loss = {} Baseline loss = {}".format(itr, np.sum(losses), np.sum(baseline_losses)))
                else:
                    print("[itr: {}] Loss = {}".format(itr, np.sum(losses)))
    except KeyboardInterrupt:
        pass
    save_weights(weights, FLAGS.weight_path)
    if FLAGS.use_baseline:
        save_weights(baseline_weights, "baseline_"+FLAGS.weight_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MAML')
    parser.add_argument('--seed', type=int, default=2, help='')
    parser.add_argument('--gradcheck', type=int, default=0, help='Run gradient check and other tests')
    parser.add_argument('--test', type=int, default=0, help='Run test on trained network to see if it works')
    parser.add_argument('--meta_lr', type=float, default=1e-3, help='Meta learning rate')
    parser.add_argument('--inner_lr', type=float, default=1e-2, help='Inner learning rate')
    parser.add_argument('--inner_bs', type=int, default=5, help='Inner batch size')
    parser.add_argument('--weight_path', type=str, default='trained_maml_weights.pkl', help='File name to save and load weights')
    parser.add_argument('--use_baseline', type=int, default=1, help='Whether to train a baseline network')
    parser.add_argument('--num_iter', type=float, default=1e4, help='Number of iterations')
    FLAGS = parser.parse_args()
    np.random.seed(FLAGS.seed)
    
    if FLAGS.gradcheck:
        gradcheck()
    elif FLAGS.test:
        test()
    else:
        train()

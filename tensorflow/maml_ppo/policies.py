import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python import layers as tf_layers
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm
from baselines.common.distributions import make_pdtype


# GLOSSARY:
# meta = outer = slow = the stuff used to optimize the meta loss
# act  = inner = fast = the stuff used to optimize the inner loss.  these get updated more frequently, because this
# makes it easier to sample and optimize the full objective 

 
# GRAPH CHUNKS
# (functions that can be called as part of defining the computational graph)
def batch_norm(inp, activation_fn, reuse, scope):
    return tf_layers.batch_norm(inp, activation_fn=activation_fn, reuse=reuse, scope=scope)

def layer_norm(inp, activation_fn, reuse, scope):
    return tf_layers.layer_norm(inp, activation_fn=activation_fn, reuse=reuse, scope=scope)

def construct_fc_weights(dim_input, dim_hidden, dim_output, prefix=''):
    """create dictionary mapping from weight strings to tf.Variables that would be used in a tf NN forward pass"""
    weights = {}
    weights[prefix+'w1'] = tf.Variable(tf.truncated_normal([dim_input, dim_hidden[0]], stddev=0.01))
    weights[prefix+'b1'] = tf.Variable(tf.zeros([dim_hidden[0]]))
    for i in range(1,len(dim_hidden)):
        weights[prefix+'w'+str(i+1)] = tf.Variable(tf.truncated_normal([dim_hidden[i-1], dim_hidden[i]], stddev=0.01))
        weights[prefix+'b'+str(i+1)] = tf.Variable(tf.zeros([dim_hidden[i]]))
    weights[prefix+'w'+str(len(dim_hidden)+1)] = tf.Variable(tf.truncated_normal([dim_hidden[-1], dim_output], stddev=0.01))
    weights[prefix+'b'+str(len(dim_hidden)+1)] = tf.Variable(tf.zeros([dim_output]))
    return weights

def forward_fc(inp, weights, dim_hidden, activ=tf.nn.relu, reuse=False, prefix=''):
    """Forward pass of a fully connected neural network. BYOW (bring your own weights)"""
    hidden = batch_norm(tf.matmul(inp, weights[prefix+'w1']) + weights[prefix+'b1'], activation_fn=activ, reuse=reuse, scope='0')
    for i in range(1,len(dim_hidden)):
        hidden = batch_norm(tf.matmul(hidden, weights[prefix+'w'+str(i+1)]) + weights[prefix+'b'+str(i+1)], activation_fn=activ, reuse=reuse, scope=str(i+1))
    return tf.matmul(hidden, weights[prefix+'w'+str(len(dim_hidden)+1)]) + weights[prefix+'b'+str(len(dim_hidden)+1)]

def construct_ppo_weights(ob_space, dim_hidden, ac_space, scope):
    """create dictionary mapping from weight strings to tf.Variables hat would be used in a tf NN forward pass"""
    with tf.variable_scope(scope):
        with tf.variable_scope('pi'):
            pi_weights = construct_fc_weights(ob_space, dim_hidden, ac_space.shape[0], prefix='pi_')
        with tf.variable_scope('vf'):
            vf_weights = construct_fc_weights(ob_space, dim_hidden, 1, prefix='vf_')
        logstd = tf.get_variable(name="logstd", shape=[1, ac_space.shape[0]], initializer=tf.zeros_initializer())
    return {**pi_weights, **vf_weights, 'logstd': logstd}

def ppo_forward(dims, weights, scope, reuse=False):
    """Full forward pass for the policy and value networks for PPO. 
    
    return action, value, neglogprob of action, and the Pd (probability distribution)
    """
    obs, dim_hidden, ac_space = dims
    with tf.variable_scope(scope, reuse=reuse):
        with tf.variable_scope("pi", reuse=reuse):
            pi = forward_fc(obs, weights, dim_hidden=dim_hidden, reuse=reuse, prefix='pi_')
        with tf.variable_scope("vf", reuse=reuse):
            vf = forward_fc(obs, weights, dim_hidden=dim_hidden, reuse=reuse, prefix='vf_')
        pdtype = make_pdtype(ac_space)
        pdparam = tf.concat([pi, pi * 0.0 + weights['logstd']], axis=1)
        pd = pdtype.pdfromflat(pdparam) # probability distribution

        a0 = pd.sample() # op for sampling from the distribution
        neglogp0 = pd.neglogp(a0) # neglogprob of that action (for gradient)
    return a0, vf, neglogp0, pd

def ppo_loss(pd, vf, sample_values, hyperparams):
    """Return op of PPO Clipped Surrogate Objective loss"""
    svs = sample_values
    hps = hyperparams

    adv = svs['returns'] - svs['oldvpreds']
    adv_mean, adv_var = tf.nn.moments(adv, axes=[0])
    adv = (adv - adv_mean) / (adv_var + 1e-8)

    neglogpac = pd.neglogp(svs['actions'])
    entropy = tf.reduce_mean(pd.entropy())

    # value prediction
    # do the clipping to prevent too much change/instability in the value function
    vpredclipped = svs['oldvpreds'] + tf.clip_by_value(vf - svs['oldvpreds'], - hps['cliprange'], hps['cliprange'])
    vf_losses1 = tf.square(vf - svs['returns'])
    vf_losses2 = tf.square(vpredclipped - svs['returns']) 
    vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2)) # times 0.5 because MSE loss
    # Compute prob ratio between old and new 
    ratio = tf.exp(svs['oldneglogpacs'] - neglogpac)
    pg_losses = -adv * ratio
    pg_losses2 = -adv * tf.clip_by_value(ratio, 1.0 - hps['cliprange'], 1.0 + hps['cliprange'])
    # Clipped Surrogate Objective (max instead of min because values are flipped)
    pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
    approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - svs['oldneglogpacs']))
    clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), hps['cliprange']))) # diagnostic: fraction of values that were clipped
    # total loss = action loss, entropy bonus, and value loss
    loss = pg_loss - entropy * hps['ent_coef'] + vf_loss * hps['vf_coef']
    return loss



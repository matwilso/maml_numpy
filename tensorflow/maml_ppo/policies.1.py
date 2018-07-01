import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python import layers as tf_layers
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm
from baselines.common.distributions import make_pdtype
from baselines.common.input import observation_input

def batch_norm(inp, activation_fn, reuse, scope):
    return tf_layers.batch_norm(inp, activation_fn=activation_fn, reuse=reuse, scope=scope)
def layer_norm(inp, activation_fn, reuse, scope):
    return tf_layers.layer_norm(inp, activation_fn=activation_fn, reuse=reuse, scope=scope)
def construct_fc_weights(dim_input, dim_hidden, dim_output):
    weights = {}
    weights['w1'] = tf.Variable(tf.truncated_normal([dim_input, dim_hidden[0]], stddev=0.01))
    weights['b1'] = tf.Variable(tf.zeros([dim_hidden[0]]))
    for i in range(1,len(dim_hidden)):
        weights['w'+str(i+1)] = tf.Variable(tf.truncated_normal([dim_hidden[i-1], dim_hidden[i]], stddev=0.01))
        weights['b'+str(i+1)] = tf.Variable(tf.zeros([dim_hidden[i]]))
    weights['w'+str(len(dim_hidden)+1)] = tf.Variable(tf.truncated_normal([dim_hidden[-1], dim_output], stddev=0.01))
    weights['b'+str(len(dim_hidden)+1)] = tf.Variable(tf.zeros([dim_output]))
    return weights
def forward_fc(inp, weights, dim_hidden, activ=tf.nn.relu, reuse=False):
    hidden = batch_norm(tf.matmul(inp, weights['w1']) + weights['b1'], activation_fn=activ, reuse=reuse, scope='0')
    for i in range(1,len(dim_hidden)):
        hidden = batch_norm(tf.matmul(hidden, weights['w'+str(i+1)]) + weights['b'+str(i+1)], activation_fn=activ, reuse=reuse, scope=str(i+1))
    return tf.matmul(hidden, weights['w'+str(len(dim_hidden)+1)]) + weights['b'+str(len(dim_hidden)+1)]


def ppo_forward(dims, weights, scope, reuse=False):
    obs, dim_hidden, ac_space = dims
    pi_weights, vf_weights, logstd = weights
    with tf.variable_scope(scope, reuse=reuse):
        with tf.variable_scope("pi", reuse=reuse):
            pi = forward_fc(obs, pi_weights, dim_hidden=dim_hidden, reuse=reuse)
        with tf.variable_scope("vf", reuse=reuse):
            vf = forward_fc(obs, vf_weights, dim_hidden=dim_hidden, reuse=reuse)
        pdtype = make_pdtype(ac_space)
        pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)
        pd = pdtype.pdfromflat(pdparam) # probability distribution

        a0 = pd.sample() # op for sampling from the distribution
        neglogp0 = pd.neglogp(a0) # neglogprob of that action (for gradient)
    return a0, vf, neglogp0, pd

def ppo_loss(pd, sample_values, hyperparams):
    svs = sample_values
    hps = hyperparams

    adv = svs['returns'] - svs['values']
    adv_mean, adv_var = tf.nn.moments(adv, axes=[1])
    adv = (adv - adv_mean) / (adv_var + 1e-8)

    neglogpac = pd.neglogp(svs['action'])
    entropy = tf.reduce_mean(pd.entropy())

    # value prediction
    # do the clipping to prevent too much change/instability in the value function
    vpredclipped = svs['oldvpred'] + tf.clip_by_value(svs['vpred'] - svs['oldvpred'], - hps['cliprange'], hps['cliprange'])
    vf_losses1 = tf.square(svs['vpred'] - svs['returns'])
    vf_losses2 = tf.square(vpredclipped - svs['returns']) 
    vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2)) # times 0.5 because MSE loss
    # Compute prob ratio between old and new 
    ratio = tf.exp(svs['oldneglogpac'] - neglogpac)
    pg_losses = -svs['adv'] * ratio
    pg_losses2 = -svs['adv'] * tf.clip_by_value(ratio, 1.0 - hps['cliprange'], 1.0 + hps['cliprange'])
    # Clipped Surrogate Objective (max instead of min because values are flipped)
    pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
    approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - svs['oldneglogpac']))
    clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), hps['cliprange']))) # diagnostic: fraction of values that were clipped
    # total loss = action loss, entropy bonus, and value loss
    loss = pg_loss - entropy * hps['ent_coef'] + vf_loss * hps['vf_coef']
    return loss



class ThinThing(object):
    def __init__(self, ac_space, noptepochs, nminibatch, seed, scope, )
        pass

    def create_placeholder_and_dataset(self):
        A = make_pdtype(ac_space).sample_placeholder([None]) # placeholder for sampled action
        R = tf.placeholder(tf.float32, [None], 'R') # Actual returns 
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [None], 'OLDNEGLOGPAC') # Old policy negative log probability (used for prob ratio calc)
        OLDVPRED = tf.placeholder(tf.float32, [None], 'OLDVPRED') # Old state-value pred

        traj_dataset = tf.data.Dataset.from_tensor_slices((self.X, R, A, OLDVPRED, OLDNEGLOGPAC))
        traj_dataset = traj_a_dataset.shuffle(SHUFFLE_BUFFER_SIZE, seed=seed).repeat(noptechos).batch(nminibatch) 
        traj_iterator = traj_a_dataset.make_initializable_iterator()
        MB_OBS, MB_R, MB_A, MB_OLDVPRED, MB_OLDNEGLOGPAC = traj_iterator.get_next()

        sample_values = dict(obs=MB_OBS, returns=MB_R, action=MB_A, oldvpred=MB_OLDVPRED, oldneglogpac=MB_OLDNEGLOGPAC)
        return sample_values


    def create_weights(self, scope):
        with tf.variable_scope(scope):
            with tf.variable_scope('pi'):
                pi_weights = construct_fc_weights(ob_space, dim_hidden, ac_space.shape[0])
            with tf.variable_scope('vf'):
                vf_weights = construct_fc_weights(ob_space, dim_hidden, 1)
            logstd = tf.get_variable(name="logstd", shape=[1, actdim], initializer=tf.zeros_initializer())
        return pi_weights, vf_weights, logstd



class Model(object):
    def __init__(self, policy, ob_space, ac_space, nbatch_act, nbatch_train, nminibatches
                nsteps, ent_coef, vf_coef, max_grad_norm, dim_hidden=[100,100], scope='model', seed=42):
        self.sess = tf.get_default_session()
        self.ac_space = ac_space
        self.op_space = ob_space
        self.dim_hidden = dim_hidden
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.X, self.processed_x = observation_input(ob_space)

        SHUFFLE_BUFFER_SIZE = nminibatches * nbatch_train
        LR = tf.placeholder(tf.float32, [], 'LR')
        CLIPRANGE = tf.placeholder(tf.float32, [], 'CLIPRANGE') # epsilon in the paper

        # WEIGHTS
        actdim = ac_space.shape[0]

        self.train_pi_weights, self.train_vf_weights, self.train_logstd = create_weights(scope='train')
        self.act_weights = self.act_pi_weights, self.act_vf_weights, self.act_logstd = create_weights(scope='act')

        self.train_vars = tf.trainable_variables('train_weights')
        self.act_vars = tf.trainable_variables('act_weights')
        sync_vars = [tf.assign(act_weight, train_weight) for act_weight, train_weight in zip(self.act_vars, self.train_vars)]

        # ACT
        dims = self.processed_x, self.dim_hidden, self.ac_space
        self.act_a, self.act_v, self.act_neglogp, self.act_pd = ppo_forward(dims, self.act_weights, scope='act', reuse=False)






    def inner_train(self):
        # FEED: traja
        pass

    def meta_train(self):
        # will run part of inner train, and then more
        # FEED: traja, trajb
        pass

    def act(self, obs):
        a, v, neglogp = self.sess.run([self.act_a, self.act_v, self.act_neglogp], {self.X:obs})
        return a, v, neglogp

    def value(self, obs):
        v = self.sess.run([self.act_v], {self.X:obs})
        return v






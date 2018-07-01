import os
import time
import joblib
import numpy as np
import os.path as osp
import tensorflow as tf
import baselines
from baselines import logger
from collections import deque
from baselines.common import explained_variance
from baselines.common.input import observation_input
from baselines.common.distributions import make_pdtype
from baselines.common.runners import AbstractEnvRunner
from policies import construct_ppo_weights, ppo_forward, ppo_loss

from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from envs.reset_vec_env import ResetValDummyVecEnv, ResetValVecNormalize


# seed for this code: https://github.com/openai/baselines/tree/master/baselines/ppo2
# paper: https://arxiv.org/abs/1707.06347 

# TODO: major refactor to make everything simpler. after we get it working
# TODO: convert the copy trajs and stuff to some class that I can just change the params I need, like scope name and number in batch
# TODO: maybe use a different X for acting, to make it a little less confusing and maybe avoid bug
# for the optimizations:
# TODO: add optimization over noptepochs
# TODO: seems like I may want to do this in a map_fn like they do in the maml implementation, instead of Dataset
# TODO: add model.load and model.save
# TODO: make a TrajInfo class to hold the returns of a sample and for feeding.  items be accesed via dictionary syntax

def dicts_to_feed(d1, d2):
    """Assign all values of d1 to the values of d2
    d1 is Dict[str, tf.placeholder]
    d2 is Dict[str, np.ndarray]

    return Dict[tf.placeholder, np.ndarray]
    """
    feed_dict = {d1[key]: d2[key] for key in d1}
    return feed_dict 
def make_traj_dataset(d):
    """Dict[str, np.ndarray] --> tf.data.Dataset of: (obs, actions, returns, oldvpreds, oldneglogpacs)"""
    return tf.data.Dataset.from_tensor_slices((d['obs'], d['actions'], d['returns'], d['oldvpreds'], d['oldneglogpacs']))
def constfn(val):
    def f(_):
        return val
    return f
def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)

class Model(object):
    """The way this class works is that the __init__ sets up the computational graph and the
    other methods are all for external use, that feed something in to run the desired parts of
    the graph"""
    def __init__(self, *, ob_space, ac_space, nbatch_act, nbatch_train, nminibatches, 
                nsteps, ent_coef, vf_coef, max_grad_norm, dim_hidden=[100,100], scope='model', seed=42):
        """This constructor sets up all the ops and the tensorflow computational graph. The other functions in
        this class are all for sess.runn-ing the various ops"""
        self.sess = tf.get_default_session()
        self.ac_space = ac_space
        self.op_space = ob_space
        ob_shape = ob_space.shape[0]
        self.dim_hidden = dim_hidden
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.X, self.processed_x = observation_input(ob_space)
        self.meta_X, self.meta_processed_x = observation_input(ob_space)

        # HYPERPARAMS
        self.CONST_HYPERPARAMS = {}
        self.CONST_HYPERPARAMS['ent_coef'] = ent_coef
        self.CONST_HYPERPARAMS['vf_coef'] = vf_coef
        # we make these feedable so we can vary them based on the step if we want
        self.DYNAMIC_HYPERPARAMS = {}
        self.DYNAMIC_HYPERPARAMS['meta_lr'] = tf.placeholder(tf.float32, [], 'META_LR')
        self.DYNAMIC_HYPERPARAMS['inner_lr'] = tf.placeholder(tf.float32, [], 'INNER_LR')
        self.DYNAMIC_HYPERPARAMS['cliprange'] = tf.placeholder(tf.float32, [], 'CLIPRANGE') # epsilon in the PPO paper
        self.ALL_HYPERPARAMS = {**self.CONST_HYPERPARAMS, **self.DYNAMIC_HYPERPARAMS}

        # INNER TRAJECTORY PLACEHOLDERS
        # TODO: turn this into a function or class to make it easier, so we don't have to repeat this for the meta
        self.IT_PHS = {}
        self.IT_PHS['obs'] = self.X
        self.IT_PHS['actions'] = make_pdtype(ac_space).sample_placeholder([None], name='a_actions') # placeholder for sampled action
        self.IT_PHS['returns'] = tf.placeholder(tf.float32, [None], 'a_returns') # Actual returns 
        self.IT_PHS['oldvpreds'] = tf.placeholder(tf.float32, [None], 'a_oldvpreds') # Old state-value pred
        self.IT_PHS['oldneglogpacs'] = tf.placeholder(tf.float32, [None], 'a_oldneglogpacs') # Old policy negative log probability (used for prob ratio calc)
        inner_traj_dataset = make_traj_dataset(self.IT_PHS)
        inner_traj_dataset = inner_traj_dataset.shuffle(buffer_size=nminibatches*nbatch_train, seed=seed)
        #inner_traj_dataset = inner_traj_dataset.repeat(noptechos) # may add this back later, after testing
        inner_traj_dataset = inner_traj_dataset.batch(nminibatches) 
        self.inner_traj_iterator = inner_traj_dataset.make_initializable_iterator()

        # META TRAJECTORY PLACEHOLDERS
        self.MT_PHS = {}
        self.MT_PHS['obs'] = self.X
        self.MT_PHS['actions'] = make_pdtype(ac_space).sample_placeholder([None], name='b_actions') # placeholder for sampled action
        self.MT_PHS['returns'] = tf.placeholder(tf.float32, [None], 'b_returns') # Actual returns 
        self.MT_PHS['oldvpreds'] = tf.placeholder(tf.float32, [None], 'b_oldvpreds') # Old state-value pred
        self.MT_PHS['oldneglogpacs'] = tf.placeholder(tf.float32, [None], 'b_oldneglogpacs') # Old policy negative log probability (used for prob ratio calc)
        meta_traj_dataset = make_traj_dataset(self.MT_PHS)
        meta_traj_dataset = meta_traj_dataset.shuffle(buffer_size=nminibatches*nbatch_train, seed=seed)
        #meta_traj_dataset = meta_traj_dataset.repeat(noptechos) # may add this back later, after testing
        meta_traj_dataset = meta_traj_dataset.batch(nminibatches) 
        self.meta_traj_iterator = meta_traj_dataset.make_initializable_iterator()

        # WEIGHTS
        # slow meta weights that only get updated after a full meta-batch
        self.slow_weights = construct_ppo_weights(ob_shape, dim_hidden, ac_space, scope='slow') 
        self.slow_vars = tf.trainable_variables(scope='slow')
        # fast act weights (these are the only ones used to act in the env. the rest are just for optimization)
        self.act_weights =  construct_ppo_weights(ob_shape, dim_hidden, ac_space, scope='act') 
        self.act_vars = tf.trainable_variables(scope='act')

        # A variable for each of the weights in slow weights, initialize to 0
        # we can pile grads up as in line 10 of the algorithm in MAML Algorithm #3.
        self.meta_grad_pile = {w : tf.Variable(initial_value=tf.zeros_like(self.slow_weights[w]), name='meta_grad_pile_'+w) for w in self.slow_weights}

        # Reset the meta grad 
        self.zero_meta_grad_ops = [tf.assign(self.meta_grad_pile[w], tf.zeros_like(self.meta_grad_pile[w])) for w in self.meta_grad_pile]
        # Sync the slow to the fast 
        self.sync_vars_ops = [tf.assign(act_weight, slow_weight) for act_weight, slow_weight in zip(self.act_vars, self.slow_vars)]

        # ACT
        act_dims = self.processed_x, self.dim_hidden, self.ac_space
        self.act_a, self.act_v, self.act_neglogp, self.act_pd = ppo_forward(act_dims, self.act_weights, scope='act', reuse=False)

        # TRAIN
        # Layout of this section:
        # This is one big line of computation.

        # this part is just for setting up the update on the act weights.
        # (may want to do several epoch runs later here, but idk how maml will perform with that)

        # INNER LOSS
        # run multiple iterations over the inner loss, updating the weights in fast weights
        fast_weights = None 
        for _ in range(5):
        #for _ in range(nminibatches):
            # 1st iter, we run with self.slow_weights, the rest will be using fast_weights
            weights = fast_weights if fast_weights is not None else self.slow_weights
            mb_obs, mb_a, mb_r, mb_oldvpred, mb_oldneglogpac = self.inner_traj_iterator.get_next()
            inner_train_dims = mb_obs, self.dim_hidden, self.ac_space
            inner_sample_values = dict(obs=mb_obs, actions=mb_a, returns=mb_r, oldvpreds=mb_oldvpred, oldneglogpacs=mb_oldneglogpac)

            inner_train_a, inner_train_vf, inner_train_neglogp, inner_train_pd = ppo_forward(inner_train_dims, weights, scope='act', reuse=True)
            inner_loss = ppo_loss(inner_train_pd, inner_train_vf, inner_sample_values, self.ALL_HYPERPARAMS)

            grads = tf.gradients(inner_loss, list(weights.values()))
            gradients = dict(zip(weights.keys(), grads))
            fast_weights = dict(zip(weights.keys(), [weights[key] - self.DYNAMIC_HYPERPARAMS['inner_lr']*gradients[key] for key in weights.keys()]))

        # capture the final act weights
        # seems like this is what we would run to update the act weights

        # Run just the inner train op.  The last step of this is to set the act_weights to be the fast weights
        # because we are about to use them to sample another trajectory.
        self.inner_train_op = [tf.assign(self.act_weights[w], fast_weights[w]) for w in fast_weights]
        self.last_inner_loss = inner_loss
        # -------------------------------------------------------------------------
        # meta half-way point
        # -------------------------------------------------------------------------
        meta_loss = 0
        #for _ in range(nminibatches):
        for _ in range(5):
            mb_obs, mb_actions, mb_returns, mb_oldvpreds, mb_oldneglogpacs = self.meta_traj_iterator.get_next()
            meta_train_dims = mb_obs, self.dim_hidden, self.ac_space
            meta_sample_values = dict(obs=mb_obs, actions=mb_actions, returns=mb_returns, oldvpreds=mb_oldvpreds, oldneglogpacs=mb_oldneglogpacs)

            # always using the same fast weights for the forward pass
            meta_train_a, meta_train_v, meta_train_neglogp, meta_train_pd = ppo_forward(meta_train_dims, fast_weights, scope='act', reuse=True)
            meta_loss += ppo_loss(meta_train_pd, meta_train_v, meta_sample_values, self.ALL_HYPERPARAMS)

        self.total_meta_loss = meta_loss

        grads = tf.gradients(meta_loss, list(self.slow_weights.values()))
        task_meta_gradients = dict(zip(self.slow_weights.keys(), grads))
        # add the new task grads in to the meta-batch grad
        self.meta_train_op = update_meta_grad_ops = [tf.assign(self.meta_grad_pile[w], self.meta_grad_pile[w] + task_meta_gradients[w]) for w in self.slow_weights]
        # zero out (reset) the meta-batch grad
        zero_meta_grad_pile_ops = [tf.assign(self.meta_grad_pile[w], tf.zeros_like(self.meta_grad_pile[w])) for w in self.meta_grad_pile]

        # zip up the grads to fit the tf.train.Optimizer API, and then apply them to update the slow weights
        meta_optimizer = tf.train.AdamOptimizer(learning_rate=self.DYNAMIC_HYPERPARAMS['meta_lr'], epsilon=1e-5)
        meta_grads_and_vars = [(self.meta_grad_pile[w], self.slow_weights[w]) for w in self.slow_weights]
        self.apply_meta_grad_ops = meta_optimizer.apply_gradients(meta_grads_and_vars, name='meta_grad_step')


        tf.global_variables_initializer().run(session=self.sess) #pylint: disable=E1101

    def act(self, obs):
        """Feed in single obs to take single action in env. Return action, value, neglogp of action"""
        a, v, neglogp = self.sess.run([self.act_a, self.act_v, self.act_neglogp], {self.X:obs})
        return a, v, neglogp

    def value(self, obs):
        v = self.sess.run([self.act_v], {self.X:obs})
        return v

    def inner_train(self, traj_sample, hyperparams):
        """inner train on 1 task in the meta-batch"""
        # reset so sampling is the same between inner and meta (important and required for meta gradient calculation to be correct)

        # Construct the feed dict from the traj sample and the hyperparams
        inner_dict = dicts_to_feed(self.IT_PHS, traj_sample)
        self.sess.run(self.inner_traj_iterator.initializer, inner_dict) 

        hype_dict = dicts_to_feed(self.DYNAMIC_HYPERPARAMS, hyperparams)
        return self.sess.run([self.inner_train_op, self.last_inner_loss], hype_dict)[1]

    def meta_train(self, inner_traj_sample, meta_traj_sample, hyperparams):
        """meta train on 1 task in the meta-batch"""

        # sync the fast and slow weights together because we are going to run through all the optimization again
        self.sess.run(self.sync_vars_ops)

        # Construct the feed dict
        inner_dict = dicts_to_feed(self.IT_PHS, inner_traj_sample)
        meta_dict = dicts_to_feed(self.MT_PHS, meta_traj_sample)
        # important to reset these. see inner_train 
        self.sess.run(self.inner_traj_iterator.initializer, inner_dict) # reset so it is same as inner train
        self.sess.run(self.meta_traj_iterator.initializer, meta_dict)

        hype_dict = dicts_to_feed(self.DYNAMIC_HYPERPARAMS, hyperparams)
        return self.sess.run([self.meta_train_op, self.total_meta_loss], feed_dict=hype_dict)[1]

    def apply_meta_grad(self, meta_lr):
        """apply the gradient update for the whole meta-batch"""
        feed_dict = {self.DYNAMIC_HYPERPARAMS['meta_lr'] : meta_lr}
        self.sess.run(self.apply_meta_grad_ops, feed_dict=feed_dict) # take a step with the meta optimizer 
        self.sess.run(self.sync_vars_ops)  # sync the act_weights so they match the new updated slow_weights
        self.sess.run(self.zero_meta_grad_ops) # zero out the meta gradient for next batch

class Runner(object):
    """Object to hold RL discounting/trace params and to run a rollout of the policy"""
    def __init__(self, *, model, nsteps, gamma, lam, render=False):
        self.model = model
        self.nsteps = nsteps
        self.gamma = gamma # discount factor
        self.lam = lam # GAE parameter used for exponential weighting of combination of n-step returns
        self.render = render

    def run(self, env):
        """Run the policy in env for the set number of steps to collect a trajectory

        Returns: Trajectory, epinfos
        """
        self.nenv = env.num_envs
        self.obs = np.zeros((self.nenv,) + env.observation_space.shape, dtype=self.model.X.dtype.name)
        self.obs[:] = env.reset()
        self.dones = [False for _ in range(self.nenv)]

        # mb = mini-batch
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        epinfos = []
        # Do a rollout of one horizon (not necessarily one ep)
        for _ in range(self.nsteps):
            actions, values, neglogpacs = self.model.act(self.obs)
            mb_obs.append(self.obs.copy().squeeze())
            mb_actions.append(actions.squeeze())
            mb_values.append(values.squeeze())
            mb_neglogpacs.append(neglogpacs.squeeze())
            mb_dones.append(self.dones)            
            self.obs[:], rewards, self.dones, infos = env.step(actions)
            if self.render: 
                env.venv.envs[0].render()
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards.squeeze())
        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs)[0]
        # discount/bootstrap off value fn (compute advantage)
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0        
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values # I don't get why they do this. Seems only for logging, since they undo it later

        def sf01(arr):
            """swap and then flatten axes 0 and 1"""
            s = arr.shape
            return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

        # obs, actions, returns, oldvpreds, neglogpacs
        trajectory = dict(obs=mb_obs, actions=mb_actions, returns=mb_returns, oldvpreds=mb_values, oldneglogpacs=mb_neglogpacs)
        return trajectory, epinfos

    # nmetaiterations = 500
    # nbatch_meta = 40
    # meta_lr = 3e-4
    # inner_lr = 0.01



def meta_learn(*, env_fn, nenvs, nsteps, num_meta_iterations, nbatch_meta, ent_coef, meta_lr, inner_lr,
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95, 
            log_interval=10, render=False, nminibatches=4, noptepochs=4, cliprange=0.2,
            save_interval=0, load_path=None, reset_fn):
    """
    Run training algo for the policy

    env: baselines.common.vec_env.VecEnv
    nsteps: T horizon in PPO paper, H in MAML paper
    total_timesteps: number of env time steps to take in all
    ent_coef: coefficient for how much to weight entropy in loss
    inner_lr: inner learning rate. function or float.  function will be passed in progress fraction (t/T) for time adaptive. float will be const
    meta_lr: meta learning rate. func or float
    vf_coef: coefficient for how much to weight value in loss
    max_grad_norm: value for determining how much to clip gradients
    gamma: discount factor
    lam: GAE lambda value (dropoff level for weighting combined n-step rewards. 0 is just 1-step TD estimate. 1 is like value baselined MC)
    nminibathces:  how many mini-batches to split data into (will divide values parameterized by nsteps)
    noptepochs:  how many optimization epochs to run. K in the PPO paper
    cliprange: epsilon in the paper. function or float.
    """

    # These allow for time-step adaptive learning rates, where pass in a function that takes in t,
    # but they default to constant functions if you pass in a float
    if isinstance(meta_lr, float): meta_lr = constfn(meta_lr)
    else: assert callable(meta_lr)
    if isinstance(inner_lr, float): inner_lr = constfn(inner_lr)
    else: assert callable(inner_lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    num_meta_iterations = int(num_meta_iterations)

    one_env = env_fn()
    ob_space = one_env.observation_space
    ac_space = one_env.action_space
    nbatch = nenvs * nsteps # number in the batch
    nbatch_train = nbatch // nminibatches # number in the minibatch for training
    nupdates = num_meta_iterations

    make_model = lambda : Model(ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train, nminibatches=nminibatches, nsteps=nsteps, max_grad_norm=max_grad_norm, scope='model', seed=42, ent_coef=ent_coef, vf_coef=vf_coef)

    if save_interval and logger.get_dir():
        import cloudpickle # cloud pickle, because writing a lamdba function (so we can call it later)
        with open(osp.join(logger.get_dir(), 'make_model.pkl'), 'wb') as fh:
            fh.write(cloudpickle.dumps(make_model))
    
    model = make_model()
    #if load_path is not None:
    #    model.load(load_path)
    runner = Runner(model=model, nsteps=nsteps, gamma=gamma, lam=lam, render=render)

    epinfobuf = deque(maxlen=100)
    tfirststart = time.time()


    for meta_update in range(num_meta_iterations):
        tstart = time.time()
        frac = 1.0 - (meta_update - 1.0) / nupdates # fraction of num of current update over num total updates
        inner_lrnow = inner_lr(frac)
        meta_lrnow = meta_lr(frac)
        cliprangenow = cliprange(frac)
        hypenow = dict(ent_coef=ent_coef, vf_coef=vf_coef, inner_lr=inner_lrnow, meta_lr=meta_lrnow, cliprange=cliprangenow)
        mblossvals = []

        for i in range(nbatch_meta):
            task_reset_val = reset_fn() 
            print(task_reset_val)
            envs = [env_fn(task_reset_val) for _ in range(nenvs)]
            env = ResetValDummyVecEnv(envs)
            env = VecNormalize(env)

            # inner sample, and train (fast step on act weights so we can take new sample)
            inner_traj_sample, meta_epinfos = runner.run(env)  # collect a trajectory of length nsteps
            inner_loss = model.inner_train(inner_traj_sample, hypenow)
            # meta sample and train (ppo loss on trajectory from fast weights w.r.t the slow weights)
            meta_traj_sample, meta_epinfos = runner.run(env)   
            meta_loss = model.meta_train(inner_traj_sample, meta_traj_sample, hypenow)

            epinfobuf.extend(meta_epinfos)
            mblossvals.append(meta_loss)
        # apply piled up gradients from meta batch
        model.apply_meta_grad(meta_lrnow)

        lossvals = np.mean(mblossvals, axis=0)
        tnow = time.time()
        fps = int(nbatch / (tnow - tstart))
        if meta_update % log_interval == 0 or meta_update == 1:
            #ev = explained_variance(values, returns)
            #logger.logkv("serial_timesteps", update*nsteps)
            logger.logkv("nupdates", meta_update)
            logger.logkv("fps", fps)
            #logger.logkv("explained_variance", float(ev))
            logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            logger.logkv('time_elapsed', tnow - tfirststart)
            #for (lossval, lossname) in zip(lossvals, model.loss_names):
            #    logger.logkv(lossname, lossval)
            logger.dumpkvs()
        #if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
        #    checkdir = osp.join(logger.get_dir(), 'checkpoints')
        #    os.makedirs(checkdir, exist_ok=True)
        #    savepath = osp.join(checkdir, '%.5i'%update)
        #    print('Saving to', savepath)
        #    model.save(savepath)
    env.close()


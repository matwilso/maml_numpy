#!/usr/bin/env python3
import argparse
from baselines import bench, logger
from envs.ant_env_rand_direc import AntEnvRandDirec
from gym.wrappers.time_limit import TimeLimit

def train(env_id, num_meta_iterations, seed, load_path, render):
    from baselines.common import set_global_seeds
    from baselines.common.vec_env.vec_normalize import VecNormalize
    import maml_ppo
    import gym
    gym.logger.set_level(40)
    import tensorflow as tf
    from envs.reset_vec_env import ResetValDummyVecEnv, ResetValVecNormalize
    ncpu = 1
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()
    def make_env():
        env = gym.make(env_id)
        env = bench.Monitor(env, logger.get_dir())
        return env

    def custom_env(goal_vel=None):
        env = AntEnvRandDirec(goal_vel=goal_vel)
        env = TimeLimit(env, max_episode_steps=1000)
        env = bench.Monitor(env, logger.get_dir(), allow_early_resets=True)
        return env

    #env = ResetValDummyVecEnv([custom_env]*5)
    #env = ResetValVecNormalize(env)

    set_global_seeds(seed)

    # in MAML paper, it looks like they are using nenvs = 40, nsteps = 200
    maml_ppo.meta_learn(env_fn=custom_env, nenvs=1, nsteps=2048, nminibatches=32, nbatch_meta=20,
        lam=0.95, gamma=0.99, noptepochs=10, log_interval=1, 
        render=render,
        save_interval=10,
        ent_coef=0.0,
        inner_lr=0.01,
        meta_lr=3e-4,
        cliprange=0.2,
        num_meta_iterations=num_meta_iterations,
        load_path=load_path, 
        reset_fn=lambda: AntEnvRandDirec.sample_goals(1))


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='Ant-v2')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num_meta_iterations', type=int, default=int(5e2))
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--render', type=int, default=0)
    args = parser.parse_args()
    logger.configure()
    train(args.env, num_meta_iterations=args.num_meta_iterations, seed=args.seed, load_path=args.load_path, render=args.render)


if __name__ == '__main__':
    main()


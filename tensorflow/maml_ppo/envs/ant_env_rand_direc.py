import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class AntEnvRandDirec(mujoco_env.MujocoEnv, utils.EzPickle):

    """Ant env to either run forwards or backwards"""
    def __init__(self, goal_vel=None):
        self.goal_direction = 0.0
        self._goal_vel = None
        mujoco_env.MujocoEnv.__init__(self, 'ant.xml', 5)
        utils.EzPickle.__init__(self)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat, 
        ])

    @staticmethod
    def sample_goals(num_goals=1):
        return -1 + 2.0*np.random.binomial(num_goals, 0.5) # -1 or 1, with probability 0.5

    def reset(self, goal_vel=None):
        self.sim.reset()
        ob = self.reset_model(goal_vel=goal_vel)
        return ob   
 
    def reset_model(self, goal_vel=None):
        # set goal_vel
        if goal_vel is not None:
            self._goal_vel = goal_vel
        elif self._goal_vel is None:
            self._goal_vel = self.sample_goals(1)
        self.goal_direction = self._goal_vel
        # reset model
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        forward_reward = self.goal_direction*(xposafter - xposbefore)/self.dt
        ctrl_cost = 0.5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward)
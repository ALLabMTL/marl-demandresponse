import gym
from gym import error, spaces, utils
from gym.utils import seeding


class FooEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        # Information (parameters) about each house + unit
        # Information about the tracked signal variation
        self.action_space = spaces.Discrete(2)
        self.state = None
        self.viewer = None
        pass

    def step(self, action):
        self.take_action(action)
        # self.status =
        reward = self.get_reward()
        ob = self.state
        # episode_over = False or True
        return ob, reward, episode_over, {}
        pass

    def take_action(self, action):
        pass

    def reset(self):
        # reinitialized states
        pass

    def render(self, mode="human"):
        pass

    def get_reward(self):
        pass

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


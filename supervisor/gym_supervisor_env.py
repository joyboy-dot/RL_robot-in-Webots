import gym
from controller import Supervisor


class SupervisorEnv(Supervisor, gym.Env):
    def step(self, action):
        """
        :param action: The agent's action
        :return: tuple, (observation, reward, is_done, info)
        """
        raise NotImplementedError

    def reset(self):
        """
        Used to reset the world to an initial state.
        :return: default observation provided by get_default_observation()
        """
        self.simulationReset()
        self.simulationResetPhysics()
        super(Supervisor, self).step(int(self.getBasicTimeStep()))
        return self.get_default_observation()

    def get_default_observation(self):
        raise NotImplementedError

    def get_observations(self, action_old, state_old):
        raise NotImplementedError

    def get_reward(self, t, state_old, state):

        raise NotImplementedError

    def is_done(self, state):

        raise NotImplementedError

    def get_info(self):

        raise NotImplementedError

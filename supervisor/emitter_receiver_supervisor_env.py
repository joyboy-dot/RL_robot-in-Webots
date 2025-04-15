from warnings import simplefilter, warn

from controller import Supervisor
from gym_supervisor_env import SupervisorEnv

class EmitterReceiverSupervisorEnv(SupervisorEnv):
    def __init__(self,
                 emitter_names=["emitter_0", "emitter_1", "emitter_2", "emitter_3"],
                 receiver_names=["receiver_0", "receiver_1", "receiver_2", "receiver_3"],
                 timestep=None):
        super().__init__()

        if timestep is None:
            self.timestep = int(self.getBasicTimeStep())    # 默认仿真计算时间步长是基本世界仿真步长
        else:
            self.timestep = timestep

        self.emitters, self.receivers = self.initialize_comms(emitter_names, receiver_names)

    def initialize_comms(self, emitter_names, receiver_names):
        emitters = [self.getDevice(name) for name in emitter_names]
        receivers = [self.getDevice(name) for name in receiver_names]

        for receiver in receivers:
            receiver.enable(self.timestep)

        return emitters, receivers

    def step(self, action, state_old, t):
        """
        action, state_old均是list
        :return: (list,float,float,_)
        """
        self.handle_emitters(action)
        if super(Supervisor, self).step(self.timestep) == -1:
            exit()
        action_old = action
        state = self.get_observations(action_old, state_old)
        reward = self.get_reward(t, state_old, state)
        return (state,
                reward,
                self.is_done(state),
                self.get_info(),
                )

    def handle_emitters(self, action):
        # 将动作指令发送给Robot
        raise NotImplementedError

    def handle_receivers(self):
        # 接收Robot的message
        raise NotImplementedError

    def get_timestep(self):
        simplefilter("once")
        warn("get_timestep is deprecated, use .timestep instead",
             DeprecationWarning)
        return self.timestep

    @property
    def timestep(self):
        return self._timestep

    @timestep.setter
    def timestep(self, value):
        """
        Setter of timestep field. Automatically converts to int as
        required by Webots.

        :param value: The new controller timestep in milliseconds
        """
        self._timestep = int(value)

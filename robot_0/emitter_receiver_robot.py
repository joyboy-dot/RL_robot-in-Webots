from warnings import simplefilter, warn

from controller import Robot


class EmitterReceiverRobot(Robot):
    def __init__(self,
                 emitter_name="emitter",
                 receiver_name="receiver",
                 timestep=None):
        super().__init__()

        if timestep is None:
            self.timestep = int(self.getBasicTimeStep())
        else:
            self.timestep = timestep

        self.emitter, self.receiver = self.initialize_comms(
            emitter_name, receiver_name)

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
        self._timestep = int(value)

    def initialize_comms(self, emitter_name, receiver_name):
        raise NotImplementedError

    def handle_emitter(self):
        raise NotImplementedError

    def handle_receiver(self):
        raise NotImplementedError

    def run(self):
        """
        This method should be called by a robot manager to run the robot.
        """
        while self.step(self.timestep) != -1:
            self.handle_receiver()
            self.handle_emitter()

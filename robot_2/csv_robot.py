from collections.abc import Iterable
import pickle
from emitter_receiver_robot import \
    EmitterReceiverRobot


class CSVRobot(EmitterReceiverRobot):
    def __init__(self,
                 emitter_name="emitter",
                 receiver_name="receiver",
                 timestep=None):
        super().__init__(emitter_name, receiver_name, timestep)

    def initialize_comms(self, emitter_name, receiver_name):
        emitter = self.getDevice(emitter_name)
        receiver = self.getDevice(receiver_name)
        receiver.enable(self.timestep)
        return emitter, receiver

    def handle_emitter(self):
        data = self.create_message()

        assert isinstance(data, Iterable), "The action object should be Iterable"

        byte_message = pickle.dumps(data)  # 将数据转换成字节流
        self.emitter.send(byte_message)

    def handle_receiver(self):
        if self.receiver.getQueueLength() > 0:
            # Receive and decode message from supervisor
            byte_message = self.receiver.getData()
            try:
                message = pickle.loads(byte_message)
                self.use_message_data(message)
            except pickle.UnpicklingError:
                print("Error in unpickling message. The data might be corrupted.")

            self.receiver.nextPacket()

    def create_message(self):
        raise NotImplementedError

    def use_message_data(self, message):
        raise NotImplementedError

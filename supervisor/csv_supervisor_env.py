import pickle
from collections.abc import Iterable
from numpy import inf
from emitter_receiver_supervisor_env import EmitterReceiverSupervisorEnv


class CSVSupervisorEnv(EmitterReceiverSupervisorEnv):
    def __init__(self,
                 emitter_names=["emitter_0", "emitter_1", "emitter_2", "emitter_3"],
                 receiver_names=["receiver_0", "receiver_1", "receiver_2", "receiver_3"],
                 timestep=None):
        super(CSVSupervisorEnv, self).__init__(emitter_names, receiver_names, timestep)

    def handle_emitters(self, action):
        # action是一个 1*14的list
        assert isinstance(action, Iterable), \
            "The action object should be Iterable"

        common_data = action[:2]
        specific_data = [action[i:i+3] for i in range(2, 14, 3)]
        # 将通用信息(x1,x2)发送给所有 emitter，并且将特定信息(x3,x4,x5)分配给每个 emitter
        for i, emitter in enumerate(self.emitters):
            message_data = common_data + specific_data[i]
            byte_message = pickle.dumps(message_data)  # 对数据进行序列化成字节流
            emitter.send(byte_message)

    def handle_receivers(self):
        """
        处理每个 receiver 设备，接收消息并返回。
        返回所有receiver接收到的消息，按顺序存储在列表中:received_message = [L1,L2,L3,L4]。
        其中L1 = [touch_value, range_value, distance_value]  L2,L3,L4 = [distance_value]
        """
        received_messages = []
        for receiver in self.receivers:
            if receiver.getQueueLength() > 0:
                try:
                    byte_message = receiver.getData()
                    message = pickle.loads(byte_message)
                    received_messages.append(message)  # 将反序列化数据添加到接收到的消息列表
                except pickle.UnpicklingError:
                    print("Error in unpickling message. The data might be corrupted.")
                receiver.nextPacket()  # 清除当前包并准备接收下一个包

            else:
                received_messages.append(None)

        return received_messages

"""robot_1 controller."""
from numpy import inf
from csv_robot import CSVRobot
import math


class RobotController(CSVRobot):
    def __init__(self):
        super().__init__()

        # 获取电机设备
        self.wheel1 = self.getDevice('wheel1')
        self.wheel2 = self.getDevice('wheel2')
        self.z1 = self.getDevice('z1')
        self.z2 = self.getDevice('z2')

        # 初始化传感器
        self.left_dis_sensor = self.getDevice('left_dis')
        self.left_dis_sensor.enable(self.timestep)

        self.right_dis_sensor = self.getDevice('right_dis')
        self.right_dis_sensor.enable(self.timestep)

        # 初始化电机设备
        self.wheel1.setPosition(inf)
        self.wheel2.setPosition(inf)
        self.wheel1.setVelocity(0.0)
        self.wheel2.setVelocity(0.0)

    def execute(self, x1, x2, x3, x4, x5):
        """
        Robot动作执行
        """
        a, b = 1, 1
        A, B = 0.6, 1
        left_speed = a * x1 + b * x2
        right_speed = a * x1 - b * x2

        # 计算Z轴运动
        p = A * (x3 + x5) / 2 * math.sin(x4 * B)

        # 设置电机和Z轴位置
        self.z1.setPosition(p)
        self.wheel1.setPosition(inf)
        self.wheel1.setVelocity(left_speed)
        self.wheel2.setPosition(inf)
        self.wheel2.setVelocity(right_speed)

    def create_message(self):
        """
        读取传感器数据并返回一维列表
        """
        left_dis_value = self.left_dis_sensor.getValue()
        right_dis_value = self.right_dis_sensor.getValue()

        return [left_dis_value, right_dis_value]

    def use_message_data(self, message):
        x1, x2, x3, x4, x5 = message[0], message[1], message[2], message[3], message[4]
        self.execute(x1, x2, x3, x4, x5)


robot_controller = RobotController()
robot_controller.run()

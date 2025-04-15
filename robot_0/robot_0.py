"""robot_0 controller."""
import math
from numpy import inf
from csv_robot import CSVRobot


class RobotController(CSVRobot):
    def __init__(self):
        super().__init__()

        # 获取电机设备
        self.wheel1 = self.getDevice('wheel1')
        self.wheel2 = self.getDevice('wheel2')
        self.z1 = self.getDevice('z1')
        self.z2 = self.getDevice('z2')

        # 初始化传感器
        self.touch_sensor = self.getDevice('touch_sensor')
        self.touch_sensor.enable(self.timestep)

        self.range_finder = self.getDevice('range_finder')
        self.range_finder.enable(self.timestep)

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
        robot动作执行
        """
        a, b = 1, 1
        A, B = 0.6, 1
        left_speed = a * x1 + b * x2
        right_speed = a * x1 - b * x2

        # 计算Z轴运动（当前不启用Z轴运动）
        p = 0 * A * (x3 + x5) / 2 * math.sin(x4 * B)  # 乘以0针对First机器人Z轴不运动，即面板不发生翻转

        # 设置电机和Z轴位置
        self.z1.setPosition(p)
        self.wheel1.setPosition(inf)
        self.wheel1.setVelocity(left_speed)
        self.wheel2.setPosition(inf)
        self.wheel2.setVelocity(right_speed)

    def create_message(self):
        """
        读取传感器上的数据，并把它拼接成一维list
        """
        touch_value = self.touch_sensor.getValue()
        range_value = self.range_finder.getRangeImage()  # 获取深度图像一维列表
        left_dis_value = self.left_dis_sensor.getValue()
        right_dis_value = self.right_dis_sensor.getValue()

        message = [touch_value] + range_value + [left_dis_value, right_dis_value]

        return message

    def use_message_data(self, message):
        x1, x2, x3, x4, x5 = message[0], message[1], message[2], message[3], message[4]
        self.execute(x1, x2, x3, x4, x5)


robot_controller = RobotController()
robot_controller.run()


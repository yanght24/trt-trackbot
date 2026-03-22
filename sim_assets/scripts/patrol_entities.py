#!/usr/bin/env python3
import math
import time

import rclpy
from rclpy.node import Node

from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState
from geometry_msgs.msg import Pose, Twist

class PatrolNode(Node):
    def __init__(self):
        super().__init__('patrol_entities')

        self.cli = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('waiting for /gazebo/set_entity_state ...')

        self.t0 = time.time()
        # 提升到 50Hz 刷新率，配合真实速度注入，达到电影级丝滑
        self.timer = self.create_timer(1.0 / 60.0, self.update)

        # 完美复刻你的红线图纸，绝对不穿模！
        self.targets = {
            # 1. 走路人：绕着老橡树 (X=10, Y=3) 转圈
            'person_walking_1': {
                'mode': 'circle',
                'cx': 10.0, 'cy': 3.0, 'z': 0.0,
                'radius': 2.5, 'speed': 0.5,
                'direction': 1 # 逆时针
            },
            # 2. 皮卡车：在右上角空白区域 (X=12, Y=-4) 转圈
            'pickup_1': {
                'mode': 'circle',
                'cx': 12.0, 'cy': -4.0, 'z': 0.0,
                'radius': 2.5, 'speed': 1.0,
                'direction': -1 # 顺时针
            },
            # 3. 女青年：在小车前方 (X=4) 左右横向巡逻
            'female_casual_1': {
                'mode': 'line_y', # 沿 Y 轴左右走
                'x': 4.0, 'y_start': 2.0, 'y_end': -2.0, 'z': 0.0,
                'speed': 0.4
            },
            # 4. 蓝车：在女青年后方、皮卡车前方的右侧 (X=7) 左右横向巡逻
            'hatchback_blue_1': {
                'mode': 'line_y',
                'x': 7.0, 'y_start': -1.0, 'y_end': -6.0, 'z': 0.0,
                'speed': 1.0
            }
        }

    def send_state(self, name, x, y, z, yaw, vx, vy, vz, vyaw):
        req = SetEntityState.Request()
        req.state = EntityState()
        req.state.name = name
        req.state.reference_frame = 'world'

        # 设置坐标 (Pose)
        req.state.pose = Pose()
        req.state.pose.position.x = float(x)
        req.state.pose.position.y = float(y)
        req.state.pose.position.z = float(z)

        # 四元数转换
        req.state.pose.orientation.z = math.sin(yaw / 2.0)
        req.state.pose.orientation.w = math.cos(yaw / 2.0)

        # 核心机密：注入真实的线速度和角速度 (Twist)，让物理引擎自行平滑过渡！
        req.state.twist = Twist()
        req.state.twist.linear.x = float(vx)
        req.state.twist.linear.y = float(vy)
        req.state.twist.linear.z = float(vz)
        req.state.twist.angular.z = float(vyaw)

        self.cli.call_async(req)

    def update(self):
        t = time.time() - self.t0

        for name, cfg in self.targets.items():
            if cfg['mode'] == 'circle':
                radius = cfg['radius']
                direction = cfg.get('direction', 1)
                omega = direction * (cfg['speed'] / radius)
                ang = omega * t
                
                # 计算位置
                x = cfg['cx'] + radius * math.cos(ang)
                y = cfg['cy'] + radius * math.sin(ang)
                yaw = ang + (math.pi / 2.0 if direction > 0 else -math.pi / 2.0)
                
                # 对位置求导，计算瞬时速度
                vx = -radius * omega * math.sin(ang)
                vy = radius * omega * math.cos(ang)
                vyaw = omega
                
                self.send_state(name, x, y, cfg['z'], yaw, vx, vy, 0.0, vyaw)

            elif cfg['mode'] == 'line_y':
                dist = abs(cfg['y_end'] - cfg['y_start'])
                period = dist / cfg['speed']
                tau = (t % (2 * period))
                
                if tau < period:
                    # 正向移动 (y_start -> y_end)
                    progress = tau / period
                    y = cfg['y_start'] + (cfg['y_end'] - cfg['y_start']) * progress
                    vy = cfg['speed'] if cfg['y_end'] > cfg['y_start'] else -cfg['speed']
                else:
                    # 反向移动 (y_end -> y_start)
                    progress = (tau - period) / period
                    y = cfg['y_end'] + (cfg['y_start'] - cfg['y_end']) * progress
                    vy = cfg['speed'] if cfg['y_start'] > cfg['y_end'] else -cfg['speed']
                
                # 调整车头/人脸朝向
                yaw = math.pi / 2.0 if vy > 0 else -math.pi / 2.0
                
                self.send_state(name, cfg['x'], y, cfg['z'], yaw, 0.0, vy, 0.0, 0.0)

def main():
    rclpy.init()
    node = PatrolNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
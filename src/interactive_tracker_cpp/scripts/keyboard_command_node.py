#!/usr/bin/env python3
"""
keyboard_command_node  —  交互式键盘控制节点

显示设计：
  • 帮助信息固定在顶部，不随目标列表滚动
  • 目标列表原地刷新（ANSI 光标定位），不产生新行
  • 刷新频率由参数 target_list_hz 控制（默认 2 Hz）
"""
import json
import select
import sys
import termios
import time
import tty

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import String

# ── 固定帮助区（始终显示在屏幕顶部）──────────────────────────────────────
HELP_LINES = [
    "╔══════════════════════════════════════════╗",
    "║     Interactive Keyboard Command Node    ║",
    "╠══════════════════════════════════════════╣",
    "║  移动控制                                ║",
    "║    w / s   前进 / 后退                   ║",
    "║    a / d   左转 / 右转                   ║",
    "║    x       停车                          ║",
    "╠══════════════════════════════════════════╣",
    "║  目标锁定                                ║",
    "║    1-9     按槽位锁定目标                ║",
    "║    u / ESC 解锁，回到手动模式            ║",
    "║    f       强制进入搜索模式              ║",
    "║    q       退出                          ║",
    "╠══════════════════════════════════════════╣",
    "║  目标列表  [槽位] id  类别  bbox高度     ║",
    "╚══════════════════════════════════════════╝",
]
HELP_HEIGHT = len(HELP_LINES)
# 目标列表最多显示行数（槽位 1-9 最多 9 个目标 + 1 行分隔）
MAX_TARGET_ROWS = 10


# ANSI 转义码工具
def _move(row: int, col: int = 1) -> str:
    return f"\033[{row};{col}H"


def _clear_line() -> str:
    return "\033[2K"


def _hide_cursor() -> str:
    return "\033[?25l"


def _show_cursor() -> str:
    return "\033[?25h"


def _clear_screen() -> str:
    return "\033[2J"


class KeyboardCommandNode(Node):
    def __init__(self):
        super().__init__("keyboard_command_node")

        # ROS2 参数（可通过命令行 --ros-args -p target_list_hz:=1.0 覆盖）
        self.declare_parameter("target_list_hz", 2.0)
        self.declare_parameter("linear_speed", 0.20)
        self.declare_parameter("angular_speed", 0.80)

        self.linear_speed: float = self.get_parameter("linear_speed").value
        self.angular_speed: float = self.get_parameter("angular_speed").value
        refresh_hz: float = max(0.1, self.get_parameter("target_list_hz").value)
        self._refresh_interval: float = 1.0 / refresh_hz

        self.manual_pub = self.create_publisher(Twist, "/manual_cmd_vel", 10)
        self.command_pub = self.create_publisher(String, "/user_command", 10)
        self.target_list_sub = self.create_subscription(
            String, "/tracker/target_list", self._target_list_callback, 10
        )

        self._target_list: list = []
        self._last_refresh: float = 0.0
        self._screen_ready: bool = False  # 首次渲染标志

    # ── 订阅回调：只缓存数据，不打印 ──────────────────────────────────────
    def _target_list_callback(self, msg: String) -> None:
        try:
            self._target_list = json.loads(msg.data)
        except Exception:
            self._target_list = []

    # ── 首次渲染：清屏 + 打印固定帮助区 ───────────────────────────────────
    def _render_initial(self, settings) -> None:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        buf = _hide_cursor() + _clear_screen()
        for i, line in enumerate(HELP_LINES):
            buf += _move(i + 1) + line
        sys.stdout.write(buf)
        sys.stdout.flush()
        tty.setraw(sys.stdin.fileno())
        self._screen_ready = True

    # ── 定频刷新目标列表区（帮助区下方，原地覆盖）────────────────────────
    def maybe_refresh(self, settings) -> None:
        now = time.monotonic()
        if now - self._last_refresh < self._refresh_interval:
            return
        self._last_refresh = now

        if not self._screen_ready:
            self._render_initial(settings)

        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)

        base_row = HELP_HEIGHT + 1  # 目标列表起始行
        buf = ""
        targets = list(self._target_list)

        if not targets:
            buf += _move(base_row) + _clear_line() + "  ( 当前无可见目标 )"
            for r in range(1, MAX_TARGET_ROWS):
                buf += _move(base_row + r) + _clear_line()
        else:
            for i, t in enumerate(targets[:9]):
                locked_tag = "  ◀ LOCKED" if t.get("locked") else ""
                line = (
                    f"  [{t['slot']}] id={t['id']:<6}"
                    f"  {t['class']:<14}"
                    f"  h={t['h']:>4}px"
                    f"{locked_tag}"
                )
                buf += _move(base_row + i) + _clear_line() + line
            # 清空多余行
            for r in range(len(targets), MAX_TARGET_ROWS):
                buf += _move(base_row + r) + _clear_line()

        # 将光标停在屏幕底部空行，避免与内容重叠
        buf += _move(base_row + MAX_TARGET_ROWS)
        sys.stdout.write(buf)
        sys.stdout.flush()
        tty.setraw(sys.stdin.fileno())

    # ── 发布接口 ───────────────────────────────────────────────────────────
    def publish_manual(self, linear_x: float, angular_z: float) -> None:
        msg = Twist()
        msg.linear.x = linear_x
        msg.angular.z = angular_z
        self.manual_pub.publish(msg)

    def publish_command(self, command: str) -> None:
        msg = String()
        msg.data = command
        self.command_pub.publish(msg)
        # 在目标列表下方单行显示最近命令（不影响固定区域）
        self._show_status(f">> 发送指令: {command}")

    def _show_status(self, text: str) -> None:
        row = HELP_HEIGHT + MAX_TARGET_ROWS + 2
        sys.stdout.write(_move(row) + _clear_line() + text)
        sys.stdout.flush()


def read_key(timeout: float = 0.05) -> str:
    dr, _, _ = select.select([sys.stdin], [], [], timeout)
    if dr:
        return sys.stdin.read(1)
    return ""


def main() -> None:
    settings = termios.tcgetattr(sys.stdin)
    rclpy.init()
    node = KeyboardCommandNode()

    try:
        tty.setraw(sys.stdin.fileno())
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)
            node.maybe_refresh(settings)

            key = read_key()
            if not key:
                continue

            if key == "q":
                break
            elif key == "w":
                node.publish_manual(node.linear_speed, 0.0)
            elif key == "s":
                node.publish_manual(-node.linear_speed, 0.0)
            elif key == "a":
                node.publish_manual(0.0, node.angular_speed)
            elif key == "d":
                node.publish_manual(0.0, -node.angular_speed)
            elif key == "x":
                node.publish_manual(0.0, 0.0)
            elif key in "123456789":
                node.publish_command(f"slot:{key}")
            elif key in ("u", " ", "\x1b"):
                node.publish_command("unlock")
            elif key == "f":
                node.publish_command("search")
    finally:
        # 恢复终端：显示光标，移到安全行
        sys.stdout.write(
            _show_cursor()
            + _move(HELP_HEIGHT + MAX_TARGET_ROWS + 4)
            + "\n"
        )
        sys.stdout.flush()
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

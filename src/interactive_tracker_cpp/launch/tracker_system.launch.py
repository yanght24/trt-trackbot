from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('interactive_tracker_cpp')
    params_file = os.path.join(pkg_share, 'config', 'tracker_params.yaml')

    tracker_node = Node(
        package='interactive_tracker_cpp',
        executable='tracker_manager_node',
        name='interactive_tracker',
        output='screen',
        parameters=[params_file],
    )

    return LaunchDescription([tracker_node])

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir     = get_package_share_directory('lidar_clustering')
    params_file = os.path.join(pkg_dir, 'config', 'params.yaml')

    input_topic_arg = DeclareLaunchArgument(
        'input_topic',
        default_value='/lidar/point_cloud',
        description='Tópico PointCloud2 publicado pelo Webots ROS2 bridge',
    )

    clustering_node = Node(
        package='lidar_clustering',
        executable='clustering_node',
        name='lidar_clustering',
        output='screen',
        emulate_tty=True,
        parameters=[
            params_file,
            {'input_topic': LaunchConfiguration('input_topic')},
        ],
    )

    return LaunchDescription([
        input_topic_arg,
        clustering_node,
    ])

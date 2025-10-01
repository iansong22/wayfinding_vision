from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='wayf_vision',
            executable='yolo_node',
            name='yolo_node',
            output='screen',
            emulate_tty=True,
            parameters=[
                {'namespace': '/camera'},
                {'rotate': True},
                {'readCameraInfo': True},
                {'base_id': 'laser'}
            ]
        ),
        Node(
            package='wayf_vision',
            executable='drspaam_node',
            name='drspaam_node',
            output='screen',
            emulate_tty=True,
            parameters=[]
        ),
        Node(
            package='wayf_vision',
            executable='kalman_node',
            name='kalman_node',
            output='screen',
            emulate_tty=True,
            parameters=[
                {'namespace': '/camera'},
                {'output_preds': False},
                {'vis_thres': -0.5},
                {'lidar_thres': -0.5},
                {'max_age': 10}
            ]
        )
    ])
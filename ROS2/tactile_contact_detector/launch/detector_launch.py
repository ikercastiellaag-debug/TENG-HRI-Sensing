from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'modo',
            default_value='bluetooth',
            description='Modo del modelo: bluetooth o cable'
        ),
        DeclareLaunchArgument(
            'puerto',
            default_value='/dev/rfcomm0',
            description='Puerto serie del dispositivo'
        ),
        Node(
            package='tactile_contact_detector',
            executable='contact_detector_node',
            name='detector_node',
            output='screen',
            parameters=[{
                'modo': LaunchConfiguration('modo'),
                'puerto': LaunchConfiguration('puerto')
            }]
        )
    ])

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction

def generate_launch_description():
    
    # Gateway (Pont vers le corps du robot)
    gateway = Node(
        package='crearobot_brain',
        executable='gateway',
        name='gateway_node',
        output='screen'
    )

    # Projection (Sortie HDMI Projecteur)
    projection = Node(
        package='crearobot_brain',
        executable='projection',
        name='projection_node',
        output='screen'
    )

    # Vision (Capture Caméra + Analyse IA)
    vision = Node(
        package='crearobot_brain',
        executable='vision',
        name='vision_node',
        output='screen'
    )

    # STT (Oreilles - Micro USB)
    stt = Node(
        package='crearobot_brain',
        executable='stt',
        name='stt_node',
        output='screen'
    )

    # Interaction (Logique de dialogue et Feedback C1/C2)
    interaction = Node(
        package='crearobot_brain',
        executable='interaction',
        name='interaction_node',
        output='screen'
    )

    return LaunchDescription([
        gateway,
        projection,
        vision,
        stt,
        interaction,
    ])
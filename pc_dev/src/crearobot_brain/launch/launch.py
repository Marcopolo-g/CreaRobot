from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction
from crearobot_brain import config

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

    # Vision (Capture Caméra)
    vision = Node(
        package='crearobot_brain',
        executable='vision',
        name='vision_node',
        output='screen'
    )

    
    # Vision (Capture Caméra) Semaine science infuse
    vision_temp = Node(
        package='crearobot_brain',
        executable='vision_temp',
        name='vision_node_temp',
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

    # Brain (Dialogue sur le dessin avec le sujet)
    brain = Node(
        package='crearobot_brain',
        executable='brain',
        name='brain_node',
        output='screen'
    )

    nodes_to_run = [gateway, stt, interaction, brain, vision]

    if config.CONDITION == "C2":
        nodes_to_run.append(projection)

    return LaunchDescription(nodes_to_run)
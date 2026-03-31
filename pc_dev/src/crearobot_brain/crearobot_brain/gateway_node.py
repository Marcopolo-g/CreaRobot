import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
import roslibpy
import json
import time

from .task_synchronizer import TaskSynchronizer
from . import config

class GatewayNode(Node):
    def __init__(self):
        super().__init__('gateway_node')

        self.ts = TaskSynchronizer()

        # Connexion au robot (ROS 1)
        self.get_logger().info('Connexion au robot QT (192.168.100.1)...')
        self.ros1_client = roslibpy.Ros(host='192.168.100.1', port=9090)
        self.ros1_client.run()

        # Topic de retour vers le robot (pour bloquer les oreilles internes si besoin)
        self.is_talking_pub = roslibpy.Topic(self.ros1_client, '/pc/is_talking', 'std_msgs/Bool')

        # Déclaration des services ROS 1 du robot
        self.talk_srv        = roslibpy.Service(self.ros1_client, '/qt_robot/behavior/talkText',  'qt_robot_interface/behavior_talk_text')
        self.gesture_srv     = roslibpy.Service(self.ros1_client, '/qt_robot/gesture/play',       'qt_robot_interface/gesture_play')
        self.emotion_srv     = roslibpy.Service(self.ros1_client, '/qt_robot/emotion/show',       'qt_robot_interface/emotion_show')
        self.voice_config_srv= roslibpy.Service(self.ros1_client, '/qt_robot/speech/config',      'qt_robot_interface/speech_config')
        self.volume_srv      = roslibpy.Service(self.ros1_client, '/qt_robot/setting/setVolume',  'qt_robot_interface/setting_setVolume')

        # Configuration de la voix dès que la connexion est prête
        self.ros1_client.on_ready(self.setup_robot_voice)

        # Subscriber ROS 2 : reçoit les ordres d'action (depuis brain_node)
        self.subscription = self.create_subscription(String, '/pc/qtaction', self.listener_callback, 10)

        self.get_logger().info("✅ Gateway Simplifiée (Commandes uniquement) Active")

    def setup_robot_voice(self):
        """ Configure la langue et le volume du robot au démarrage """
        try:
            self.get_logger().info("Configuration vocale du robot...")
            req_speech = roslibpy.ServiceRequest({
                'language': config.VOICE_LANG,
                'pitch'   : config.VOICE_PITCH,
                'speed'   : config.VOICE_SPEED
            })
            self.voice_config_srv.call(req_speech)

            req_volume = roslibpy.ServiceRequest({'volume': int(config.VOICE_VOLUME)})
            self.volume_srv.call(req_volume)
            self.get_logger().info("--- ROBOT PRÊT À PARLER ---")
        except Exception as e:
            self.get_logger().error(f"Erreur config voix : {e}")

    def listener_callback(self, msg):
        """ Reçoit [Geste, Emotion, Texte] et l'exécute sur le robot """
        try:
            data = json.loads(msg.data)
            geste, emotion, texte = data[0], data[1], data[2]

            # On signale au robot qu'il parle (pour éviter qu'il s'écoute lui-même)
            self.is_talking_pub.publish(roslibpy.Message({'data': True}))
            
            self.get_logger().info(f'Action : {geste} | {emotion} | "{texte}"')

            # Synchronisation des tâches
            tasks = [
                (0, lambda: self.talk_srv.call(roslibpy.ServiceRequest({'message': texte}))),
                (0, lambda: self.emotion_srv.call(roslibpy.ServiceRequest({'name': 'QT/' + emotion}))),
                (0, lambda: self.gesture_srv.call(roslibpy.ServiceRequest({'name': 'QT/' + geste, 'speed': 0})))
            ]
            self.ts.sync(tasks)

            self.is_talking_pub.publish(roslibpy.Message({'data': False}))

        except Exception as e:
            # Si c'est juste une chaîne simple, on fait juste parler le robot
            self.get_logger().warn(f"Format JSON non détecté, exécution texte simple.")
            self.talk_srv.call(roslibpy.ServiceRequest({'message': msg.data}))

def main(args=None):
    rclpy.init(args=args)
    node = GatewayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.ros1_client.terminate()
        node.destroy_node()
        rclpy.shutdown()
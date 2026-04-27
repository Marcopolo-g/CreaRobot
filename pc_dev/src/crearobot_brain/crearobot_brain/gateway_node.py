import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Float64MultiArray
import roslibpy
import json
import time
import os

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

        # Topic ROS 1 pour le contrôle de la tête 
        self.head_pub_ros1 = roslibpy.Topic(self.ros1_client, '/qt_robot/head_position/command', 'std_msgs/Float64MultiArray')

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

        # Subscriber ROS 2 pour la position de la tête 
        self.head_sub = self.create_subscription(Float64MultiArray, '/pc/head_position/command', self.head_callback, 10)

        self.get_logger().info("Gateway Active")

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
        """ Reçoit [Geste, Emotion, Texte] et traite le texte seul en priorité """
        try:
            data = json.loads(msg.data)
            geste, emotion, texte = data[0], data[1], data[2]
            
            # Nettoyage des variables
            has_text = texte and texte.strip() != ""
            no_geste = not geste or geste.lower() == "none"
            no_emotion = not emotion or emotion.lower() == "none"

            # Texte pur (pas de geste ni d'emotion). Envoi direct sans sync.
            if has_text and no_geste and no_emotion:
                self.get_logger().info(f'Direct Talk : "{texte}"')
                self.talk_srv.call(roslibpy.ServiceRequest({'message': texte}))
                return

            # Animation pure (pas de texte). On sync pour les gestes, bouche reste fermee.
            if not has_text:
                tasks = []
                if not no_emotion:
                    tasks.append((0, lambda: self.emotion_srv.call(roslibpy.ServiceRequest({'name': 'QT/' + emotion}))))
                if not no_geste:
                    tasks.append((0, lambda: self.gesture_srv.call(roslibpy.ServiceRequest({'name': 'QT/' + geste, 'speed': 0}))))
                
                if tasks:
                    self.get_logger().info("Animation seule (Bouche fermee)")
                    self.ts.sync(tasks)
                return

            # Action combinee (Texte + Geste/Emotion). Orchestration complete.
            self.get_logger().info(f'Action Synchro : {geste} | {emotion} | "{texte}"')
            tasks = [
                (0, lambda: self.talk_srv.call(roslibpy.ServiceRequest({'message': texte}))),
                (0, lambda: self.emotion_srv.call(roslibpy.ServiceRequest({'name': 'QT/' + emotion}))),
                (0, lambda: self.gesture_srv.call(roslibpy.ServiceRequest({'name': 'QT/' + geste, 'speed': 0})))
            ]
            self.ts.sync(tasks)

        except Exception as e:
            # Fallback simple si le format n'est pas bon
            if msg.data and msg.data.strip() != "":
                self.talk_srv.call(roslibpy.ServiceRequest({'message': msg.data}))
        
    def head_callback(self, msg):
        """ Reçoit la position [Yaw, Pitch] en ROS 2 et l'envoie au robot en ROS 1 """
        try:
            # On prépare le message pour ROS 1 (dictionnaire pour roslibpy)
            ros1_msg = roslibpy.Message({'data': list(msg.data)})
            self.head_pub_ros1.publish(ros1_msg)
        except Exception as e:
            self.get_logger().error(f"Erreur envoi tête ROS 1 : {e}")

def main(args=None):
    rclpy.init(args=args)
    node = GatewayNode() 

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()
        
        # Sortie immediate pour eviter la pollution des logs ROS 2
        os._exit(0)
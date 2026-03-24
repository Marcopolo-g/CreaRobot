import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, UInt8MultiArray
from sensor_msgs.msg import Image as ROS2Image
import roslibpy
import json
import time
import base64

from ..task_synchronizer import TaskSynchronizer
from .. import config

class GatewayNode(Node):
    def __init__(self):
        super().__init__('gateway_node_test')

        self.ts = TaskSynchronizer()

        # Connexion au robot ROS 1
        self.get_logger().info('Connexion au robot (192.168.100.1:9090)...')
        self.ros1_client = roslibpy.Ros(host='192.168.100.1', port=9090)
        
        try:
            self.ros1_client.run()
        except Exception as e:
            self.get_logger().error(f"Impossible de se connecter au bridge ROS 1 : {e}")

        # --- PUBLISHERS (PC -> Robot ou PC -> PC) ---
        self.speech_pub_ros2 = self.create_publisher(String, '/pc/user_speech', 10)
        self.image_pub_ros2  = self.create_publisher(ROS2Image, '/pc/camera/image', 10)
        self.is_talking_pub  = roslibpy.Topic(self.ros1_client, '/pc/is_talking', 'std_msgs/Bool')

        # --- SERVICES ROBOT ---
        self.talk_srv         = roslibpy.Service(self.ros1_client, '/qt_robot/behavior/talkText',  'qt_robot_interface/behavior_talk_text')
        self.gesture_srv      = roslibpy.Service(self.ros1_client, '/qt_robot/gesture/play',       'qt_robot_interface/gesture_play')
        self.emotion_srv      = roslibpy.Service(self.ros1_client, '/qt_robot/emotion/show',       'qt_robot_interface/emotion_show')
        self.voice_config_srv = roslibpy.Service(self.ros1_client, '/qt_robot/speech/config',      'qt_robot_interface/speech_config')
        self.volume_srv       = roslibpy.Service(self.ros1_client, '/qt_robot/setting/setVolume',  'qt_robot_interface/setting_setVolume')

        # --- BRIDGE CAMÉRA ---
        # throttle_rate en ms (10000 = 1 image toutes les 10s)
        self.cam_topic_ros1 = roslibpy.Topic(self.ros1_client, '/camera/color/image_raw', 'sensor_msgs/Image', throttle_rate=5000, queue_length=1)
        self.cam_topic_ros1.subscribe(self.bridge_camera_callback)

        # --- CONFIGURATION STT ---
        if config.STT_MODE == "google":
            self.stt_topic_ros1 = roslibpy.Topic(self.ros1_client, '/pc/user_speech', 'std_msgs/String')
            self.stt_topic_ros1.subscribe(self.bridge_text_callback)
        elif config.STT_MODE == "whisper":
            self.pub_audio_ros2 = self.create_publisher(UInt8MultiArray, '/pc/raw_audio', 100)
            self.audio_topic_ros1 = roslibpy.Topic(self.ros1_client, '/pc/raw_audio', 'std_msgs/UInt8MultiArray')
            self.audio_topic_ros1.subscribe(self.bridge_audio_callback)

        # Config auto au démarrage
        self.ros1_client.on_ready(self.setup_robot_voice)

        # Commande depuis ROS 2
        self.subscription = self.create_subscription(String, '/pc/qtaction', self.listener_callback, 10)

    def bridge_camera_callback(self, message):
        """ Décodage et bridge de l'image ROS 1 (Base64) vers ROS 2 """
        try:
            ros2_img = ROS2Image()
            ros2_img.header.stamp = self.get_clock().now().to_msg()
            ros2_img.header.frame_id = "camera_link"
            ros2_img.height = message['height']
            ros2_img.width = message['width']
            ros2_img.encoding = message['encoding']
            ros2_img.is_bigendian = message['is_bigendian']
            ros2_img.step = message['step']
            ros2_img.data = list(base64.b64decode(message['data']))
            self.image_pub_ros2.publish(ros2_img)
        except Exception as e:
            self.get_logger().error(f"Erreur Bridge Caméra : {e}")

    def bridge_text_callback(self, message):
        """ Mode Google : Bridge texte ROS 1 -> ROS 2 """
        ros2_msg = String()
        ros2_msg.data = message['data']
        self.speech_pub_ros2.publish(ros2_msg)

    def bridge_audio_callback(self, message):
        """ Mode Whisper : Bridge audio brut ROS 1 -> ROS 2 """
        msg = UInt8MultiArray()
        msg.data = [int(b) & 0xFF for b in base64.b64decode(message['data'])]
        self.pub_audio_ros2.publish(msg)

    def setup_robot_voice(self):
        """ Configure la voix et le volume au démarrage """
        self.get_logger().info("Configuration Voix & Volume...")
        try:
            self.voice_config_srv.call(roslibpy.ServiceRequest({
                'language': config.VOICE_LANG,
                'pitch': config.VOICE_PITCH,
                'speed': config.VOICE_SPEED
            }))
            self.volume_srv.call(roslibpy.ServiceRequest({'volume': int(config.VOICE_VOLUME)}))
            self.get_logger().info("--- GATEWAY OPÉRATIONNELLE ---")
        except Exception as e:
            self.get_logger().warn(f"Échec setup voix : {e}")

    def listener_callback(self, msg):
        """ Reçoit [geste, emotion, texte] de ROS 2 et l'exécute sur le robot """
        try:
            data = json.loads(msg.data)
            geste, emotion, texte = data[0], data[1], data[2]

            # Bloque le micro (AEC logiciel)
            self.is_talking_pub.publish(roslibpy.Message({'data': True}))
            
            # Attendre un tout petit peu que l'ordre de lock arrive au robot
            time.sleep(0.05)

            self.get_logger().info(f'Robot dit : "{texte}" ({geste}/{emotion})')

            tasks = [
                (0, lambda: self.talk_srv.call(roslibpy.ServiceRequest({'message': texte}))),
                (0, lambda: self.emotion_srv.call(roslibpy.ServiceRequest({'name': 'QT/' + emotion}))),
                (0, lambda: self.gesture_srv.call(roslibpy.ServiceRequest({'name': 'QT/' + geste, 'speed': 0})))
            ]
            self.ts.sync(tasks)

            # Relance le micro
            self.is_talking_pub.publish(roslibpy.Message({'data': False}))

        except Exception as e:
            # Si c'est juste un string (pas JSON), on le fait juste parler
            self.get_logger().info(f'Parole simple : {msg.data}')
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
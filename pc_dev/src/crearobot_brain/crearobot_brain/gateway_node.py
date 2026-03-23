import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from sensor_msgs.msg import Image as ROS2Image
import roslibpy
import json
import time
import base64

from .task_synchronizer import TaskSynchronizer
from . import config

class GatewayNode(Node):
    def __init__(self):
        super().__init__('gateway_node')

        self.ts = TaskSynchronizer()

        # Connexion au robot
        self.get_logger().info('Tentative de connexion au robot (ROS 1)')
        self.ros1_client = roslibpy.Ros(host='192.168.100.1', port=9091)
        self.ros1_client.run()

        # Publisher vers ROS 2 (brain_node)
        self.speech_pub_ros2 = self.create_publisher(String, '/pc/user_speech', 10)

        # Publisher "is_talking" vers le robot (pour ears.py)
        self.is_talking_pub = roslibpy.Topic(self.ros1_client, '/pc/is_talking', 'std_msgs/Bool')

        # Publisher Image pour le camera_node
        self.image_pub_ros2  = self.create_publisher(ROS2Image, '/pc/camera/image', 10)

        # Déclaration des services robot
        self.talk_srv        = roslibpy.Service(self.ros1_client, '/qt_robot/behavior/talkText',  'qt_robot_interface/behavior_talk_text')
        self.gesture_srv     = roslibpy.Service(self.ros1_client, '/qt_robot/gesture/play',       'qt_robot_interface/gesture_play')
        self.emotion_srv     = roslibpy.Service(self.ros1_client, '/qt_robot/emotion/show',       'qt_robot_interface/emotion_show')
        self.voice_config_srv= roslibpy.Service(self.ros1_client, '/qt_robot/speech/config',      'qt_robot_interface/speech_config')
        self.volume_srv      = roslibpy.Service(self.ros1_client, '/qt_robot/setting/setVolume',  'qt_robot_interface/setting_setVolume')

        # Bridge Caméra ROS 1 -> ROS 2 
        self.cam_topic_ros1 = roslibpy.Topic(self.ros1_client, '/camera/color/image_raw', 'sensor_msgs/Image', throttle_rate=10000, queue_length=1)
        self.cam_topic_ros1.subscribe(self.bridge_camera_callback)
        self.get_logger().info("Bridge Caméra ROS1 → ROS2 actif")

        # -------------------------------------------------------
        # MODE STT
        # -------------------------------------------------------
        if config.STT_MODE == "google":
            self.get_logger().info("Mode STT : GOOGLE (texte depuis le robot)")
            self._setup_google_mode()
        elif config.STT_MODE == "whisper":
            self.get_logger().info("Mode STT : WHISPER (audio brut depuis le robot)")
            self._setup_whisper_mode()
        else:
            self.get_logger().error(f"STT_MODE inconnu : '{config.STT_MODE}'. Valeurs valides : 'google' ou 'whisper'")

        # Configuration automatique du robot dès la connexion établie
        self.ros1_client.on_ready(self.setup_robot_voice)

        # Subscriber ROS 2 : reçoit les actions depuis brain_node
        self.subscription = self.create_subscription(String, '/pc/qtaction', self.listener_callback, 10)

    # -------------------------------------------------------
    # CALLBACK BRIDGE CAMÉRA
    # -------------------------------------------------------
    def bridge_camera_callback(self, message):
        """ Reçoit l'image du robot (Base64), la décode et la republie en ROS 2 """
        try:
            # Création du message ROS 2
            ros2_img = ROS2Image()
            
            # On remplit les entêtes
            ros2_img.header.stamp = self.get_clock().now().to_msg()
            ros2_img.header.frame_id = "camera_link"
            ros2_img.height = message['height']
            ros2_img.width = message['width']
            ros2_img.encoding = message['encoding'] # Souvent 'rgb8'
            ros2_img.is_bigendian = message['is_bigendian']
            ros2_img.step = message['step']

            # Décodage de la donnée Base64 en bytes, puis en liste d'entiers pour ROS 2
            raw_data = base64.b64decode(message['data'])
            ros2_img.data = list(raw_data)

            # On envoie vers le camera_node
            self.image_pub_ros2.publish(ros2_img)

        except Exception as e:
            self.get_logger().error(f"Erreur Bridge Caméra : {e}")

    # -------------------------------------------------------
    # SETUP DES DEUX MODES
    # -------------------------------------------------------

    def _setup_google_mode(self):
        """
        MODE GOOGLE
        Le robot (ears.py) fait le STT lui-même et publie du TEXTE sur /pc/user_speech (ROS 1).
        La gateway se contente de bridger ce texte vers ROS 2.
        """
        self.stt_topic_ros1 = roslibpy.Topic(self.ros1_client, '/pc/user_speech', 'std_msgs/String')
        self.stt_topic_ros1.subscribe(self.bridge_text_callback)
        self.get_logger().info("Bridge texte ROS1 → ROS2 actif sur /pc/user_speech")

    def _setup_whisper_mode(self):
        """
        MODE WHISPER
        Le robot (ears_v2.py) envoie de l'AUDIO BRUT sur /pc/raw_audio (ROS 1).
        La gateway bridge cet audio vers ROS 2 où stt_node.py (Whisper) fait la transcription.
        Le texte final arrive quand même sur /pc/user_speech mais côté ROS 2 (publié par stt_node).
        """

        from std_msgs.msg import UInt8MultiArray  
        self.pub_audio_ros2 = self.create_publisher(UInt8MultiArray, '/pc/raw_audio', 100)
        self.audio_topic_ros1 = roslibpy.Topic(
            self.ros1_client, '/pc/raw_audio', 'std_msgs/UInt8MultiArray')
        self.audio_topic_ros1.subscribe(self.bridge_audio_callback)
        self.get_logger().info("Bridge audio brut ROS1 → ROS2 actif sur /pc/raw_audio")


    # -------------------------------------------------------
    # CALLBACKS DES DEUX MODES
    # -------------------------------------------------------

    def bridge_text_callback(self, message):
        """MODE GOOGLE — reçoit du texte depuis ROS 1 et le publie en ROS 2"""

        t_received = time.time()  # ← heure d'arrivée depuis le robot
        
        ros2_msg = String()
        ros2_msg.data = message['data']
        self.speech_pub_ros2.publish(ros2_msg)
        
        t_published = time.time()
        self.get_logger().info(
            f"[GOOGLE] Texte bridgé : '{message['data']}' | "
            f"Bridge ROS1→ROS2 : {(t_published - t_received)*1000:.1f}ms")

    def bridge_audio_callback(self, message):
        """MODE WHISPER — reçoit de l'audio brut depuis ROS 1 et le publie en ROS 2"""
        import base64
        from std_msgs.msg import UInt8MultiArray

        msg = UInt8MultiArray()
        raw = base64.b64decode(message['data'])
        msg.data = [int(b) & 0xFF for b in raw]
        self.pub_audio_ros2.publish(msg)
    

    ''' STT MODES


    def _setup_google_mode(self):
        self.stt_topic_ros1 = roslibpy.Topic(self.ros1_client, '/pc/user_speech', 'std_msgs/String')
        self.stt_topic_ros1.subscribe(self.bridge_text_callback)

    def _setup_whisper_mode(self):
        from std_msgs.msg import UInt8MultiArray  
        self.pub_audio_ros2 = self.create_publisher(UInt8MultiArray, '/pc/raw_audio', 100)
        self.audio_topic_ros1 = roslibpy.Topic(self.ros1_client, '/pc/raw_audio', 'std_msgs/UInt8MultiArray')
        self.audio_topic_ros1.subscribe(self.bridge_audio_callback)

    def bridge_text_callback(self, message):
        ros2_msg = String()
        ros2_msg.data = message['data']
        self.speech_pub_ros2.publish(ros2_msg)

    def bridge_audio_callback(self, message):
        from std_msgs.msg import UInt8MultiArray
        msg = UInt8MultiArray()
        raw = base64.b64decode(message['data'])
        msg.data = [int(b) & 0xFF for b in raw]
        self.pub_audio_ros2.publish(msg)

    '''



    # -------------------------------------------------------
    # CONFIGURATION AUTOMATIQUE DU ROBOT
    # -------------------------------------------------------
    
    def setup_robot_voice(self):
        self.get_logger().info("Configuration du robot en cours...")

        req_speech = roslibpy.ServiceRequest({
            'language': config.VOICE_LANG,
            'pitch'   : config.VOICE_PITCH,
            'speed'   : config.VOICE_SPEED
        })
        self.voice_config_srv.call(req_speech)

        req_volume = roslibpy.ServiceRequest({'volume': int(config.VOICE_VOLUME)})
        self.volume_srv.call(req_volume)

        self.get_logger().info(
            f"--- GATEWAY PRÊTE (mode STT : {config.STT_MODE}) ---")

    # -------------------------------------------------------
    # EXECUTION DES ACTIONS SUR LE ROBOT
    # -------------------------------------------------------

    def listener_callback(self, msg):
        try:
            data   = json.loads(msg.data)
            geste  = data[0]
            emotion= data[1]
            texte  = data[2]

            # Signale aux oreilles de ne plus écouter
            self.get_logger().info('LOCK : Micro robot désactivé.')
            self.is_talking_pub.publish(roslibpy.Message({'data': True}))
            time.sleep(0.1)

            self.get_logger().info(f'Exécution : {geste} + {emotion} + "{texte}"')

            tasks = [
                (0, lambda: self.talk_srv.call(
                    roslibpy.ServiceRequest({'message': texte}))),
                (0, lambda: self.emotion_srv.call(
                    roslibpy.ServiceRequest({'name': 'QT/' + emotion}))),
                (0, lambda: self.gesture_srv.call(
                    roslibpy.ServiceRequest({'name': 'QT/' + geste, 'speed': 0})))
            ]
            self.ts.sync(tasks)

            self.is_talking_pub.publish(roslibpy.Message({'data': False}))
            self.get_logger().info('UNLOCK : Hyppolite ré-écoute.')

        except Exception as e:
            self.get_logger().warn(f'Message simple (pas JSON) : {msg.data}')
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
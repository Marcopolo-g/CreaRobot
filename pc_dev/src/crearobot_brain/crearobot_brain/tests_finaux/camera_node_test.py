import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import numpy as np
from cv_bridge import CvBridge 
from .. import config

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node_test')
        
        self.bridge = CvBridge()
        self.latest_msg = None
        
        # On s'abonne au flux d'images qui vient de la Gateway
        self.subscription = self.create_subscription(Image, '/pc/camera/image', self.image_callback, 1)
        
        # On crée un Timer qui se déclenche selon ta variable config
        interval = config.PHOTO_INTERVAL
        self.timer = self.create_timer(interval, self.take_photo_callback)
        
        self.get_logger().info(f'Nœud Caméra prêt. Photo prévue toutes les {interval} secondes.')

    def image_callback(self, msg):
        """ Stocke simplement le dernier message reçu en mémoire """
        self.latest_msg = msg

    def take_photo_callback(self):
        """ Appelé par le timer pour sauvegarder la dernière image reçue """
        if self.latest_msg is None:
            self.get_logger().warn("Timer déclenché mais aucune image reçue du robot pour l'instant.")
            return

        try:
            # Conversion du message ROS2 (rgb8) vers OpenCV (bgr8 pour le fichier)
            # On utilise 'bgr8' car OpenCV enregistre en BGR par défaut
            cv_image = self.bridge.imgmsg_to_cv2(self.latest_msg, desired_encoding='bgr8')
            
            # Sauvegarde de l'image sur le disque
            filename = config.SAVE_PATH
            cv2.imwrite(filename, cv_image)
            
            self.get_logger().info(f'Photo enregistrée : {filename}')
            
            # Ici, tu pourras plus tard ajouter l'appel à ton LLM
            # self.send_to_llm(filename)

        except Exception as e:
            self.get_logger().error(f'Erreur lors de la capture : {e}')

def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import cv2
import base64

from . import config

class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')
        
        # --- PUBLISHER ---
        # On envoie l'image en string Base64 au BrainNode
        self.image_b64_pub = self.create_publisher(String, '/pc/vision/image_raw_b64', 10)
        
        # --- SUBSCRIPTION ---
        self.trigger_sub = self.create_subscription(String, '/pc/camera/trigger', self.handle_trigger, 10)
            
        self.get_logger().info("Vision Node prêt")

    def handle_trigger(self, msg):
        if msg.data == "ANALYZE_IMAGE":
            self.get_logger().info("Trigger reçu, capture de l'image...")
            
            # Ouverture de la caméra 
            cap = cv2.VideoCapture(2) 

            if not cap.isOpened():
                self.get_logger().error("ÉCHEC : Caméra inaccessible (vérifie l'index ou le branchement).")
                return

            # Vidage du buffer pour garantir une image synchronisée
            for _ in range(5):
                cap.read()
            
            ret, frame = cap.read()
            cap.release() # On libère la caméra immédiatement

            if ret:
                cv2.imwrite(config.IMAGE_PATH, frame)
                
                self.send_image_to_brain(frame)
            else:
                self.get_logger().error("ÉCHEC : Capture impossible.")

    def send_image_to_brain(self, frame):
        try:
            # Encodage en JPG
            _, buffer = cv2.imencode('.jpg', frame)
            
            # Conversion en Base64 pour le transport via String ROS 2
            base64_str = base64.b64encode(buffer).decode('utf-8')
            
            # Publication
            msg = String()
            msg.data = base64_str
            self.image_b64_pub.publish(msg)
            
            self.get_logger().info("Image envoyée sur /pc/vision/image_raw_b64")
            
        except Exception as e:
            self.get_logger().error(f"Erreur lors de l'encodage/envoi : {e}")

def main():
    rclpy.init()
    node = VisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
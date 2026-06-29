import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import cv2
import base64
import numpy as np
import mss # bibliothèque de capture d'écran
import os

from . import config

class VisionScreenNode(Node):
    def __init__(self):
        super().__init__('vision_screen_node')
        
        # Publisher pour le Brain
        self.image_b64_pub = self.create_publisher(String, '/pc/vision/image_raw_b64', 10)
        
        # Écoute du trigger
        self.trigger_sub = self.create_subscription(String, '/pc/camera/trigger', self.handle_trigger, 10)
            
        self.get_logger().info("Vision Screen Node prêt (Mode HDMI)")

    def handle_trigger(self, msg):
        if msg.data == "ANALYZE_IMAGE":
            self.get_logger().info("Trigger reçu, capture de l'écran HDMI...")
            
            try:
                with mss.mss() as sct:
                    # Mettre le 2eme ecran en HMDI sur Ecran 2
                    monitor_number = 2 
                    
                    if monitor_number >= len(sct.monitors):
                        self.get_logger().error(f"Écran {monitor_number} introuvable.")
                        return

                    monitor = sct.monitors[monitor_number]
                    
                    # Capture de l'ecran
                    screenshot = sct.grab(monitor)
                    
                    # Conversion en format OpenCV (BGR)
                    frame = np.array(screenshot)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                    # Sauvegarde locale
                    cv2.imwrite(config.IMAGE_PATH, frame)
                    
                    self.send_image_to_brain(frame)
                    
            except Exception as e:
                self.get_logger().error(f"Échec de la capture d'écran : {e}")

    def send_image_to_brain(self, frame):
        try:
            # Encodage JPG
            _, buffer = cv2.imencode('.jpg', frame)
            
            # Passage en Base64
            base64_str = base64.b64encode(buffer).decode('utf-8')
            
            msg = String()
            msg.data = base64_str
            self.image_b64_pub.publish(msg)
            
            self.get_logger().info("Capture d'écran envoyée au Brain")
            
        except Exception as e:
            self.get_logger().error(f"Erreur encodage : {e}")

def main(args=None):
    rclpy.init(args=args)
    node = VisionScreenNode() 

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()
        
        os._exit(0)
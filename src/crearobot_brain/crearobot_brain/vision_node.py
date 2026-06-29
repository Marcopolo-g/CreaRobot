import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from rclpy.qos import QoSProfile, DurabilityPolicy
import cv2
import base64
import os
import json
import datetime

from . import config

class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')

        self.current_phase = "UNKNOWN"
        self.current_tour  = 0
        self._photo_index  = 0
        self._photos_dir   = None  # fixé quand on reçoit /pc/session_dir

        self.image_b64_pub = self.create_publisher(String, '/pc/vision/image_raw_b64', 10)
        self.trigger_sub   = self.create_subscription(String, '/pc/camera/trigger',  self.handle_trigger,  10)
        self.phase_sub     = self.create_subscription(String, '/pc/phase_control',   self.phase_callback,  10)

        # S'abonne au chemin de session publié par brain_node (latched)
        _latched_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, '/pc/session_dir', self._on_session_dir, _latched_qos)

        self.get_logger().info("Vision Node prêt — en attente du dossier de session...")

    def _on_session_dir(self, msg):
        if self._photos_dir is not None:
            return  # déjà reçu
        session_dir = msg.data
        self._photos_dir = os.path.join(session_dir, "dessins")
        os.makedirs(self._photos_dir, exist_ok=True)
        self.get_logger().info(f"Dossier photos : {self._photos_dir}")

    def phase_callback(self, msg):
        try:
            data = json.loads(msg.data)
            self.current_phase = data.get("phase", "UNKNOWN")
            self.current_tour  = data.get("tour", 0)
        except Exception:
            pass

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
                # Sauvegarde horodatée dans le dossier de session
                if self._photos_dir is not None:
                    self._photo_index += 1
                    ts    = datetime.datetime.now().strftime("%H-%M-%S")
                    phase = self.current_phase.replace("START_", "").lower()
                    fname = f"{self._photo_index:02d}_{phase}_tour{self.current_tour}_{ts}.jpg"
                    cv2.imwrite(os.path.join(self._photos_dir, fname), frame)
                    self.get_logger().info(f"Photo sauvegardée : {fname}")
                else:
                    self.get_logger().warn("Dossier de session pas encore reçu, photo non sauvegardée.")

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

def main(args=None):
    rclpy.init(args=args)
    node = VisionNode() 

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()
        
        os._exit(0)
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from rclpy.qos import QoSProfile, DurabilityPolicy
from cv_bridge import CvBridge
import cv2
import numpy as np
import subprocess
import os
import datetime

class ProjectionNode(Node):
    def __init__(self):
        super().__init__('projection_node')
        self.bridge = CvBridge()

        # --- CONFIGURATION ---
        self.projector_x_offset = 3072  # largeur de l'écran principal (3072x1728)
        self.window_name = "Projection_QT"
        self.current_image = None
        self._generated_dir = None
        self._image_index = 0

        # --- DÉPLOIEMENT FENÊTRE ---
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        if self.is_projector_connected():
            self.get_logger().info("Projecteur détecté. Déploiement plein écran...")
            cv2.moveWindow(self.window_name, self.projector_x_offset, 0)
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            self.get_logger().warning("Projecteur NON DÉTECTÉ — fenêtre sur écran principal.")

        # Subscriber image
        self.subscription = self.create_subscription(Image, '/pc/projector/image', self.image_callback, 10)

        # Chemin de session (latché) publié par brain_node
        _latched_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, '/pc/session_dir', self._on_session_dir, _latched_qos)

        # Timer pour rafraîchir l'image (20 FPS)
        self.gui_timer = self.create_timer(0.05, self.update_gui)

        self.get_logger().info(f"Projection initialisée sur le projecteur (Offset: {self.projector_x_offset})")

    def _on_session_dir(self, msg):
        if self._generated_dir is not None:
            return
        self._generated_dir = os.path.join(msg.data, "dessins_generes")
        os.makedirs(self._generated_dir, exist_ok=True)
        self.get_logger().info(f"Dossier dessins générés : {self._generated_dir}")

    def is_projector_connected(self):
        """ Vérifie via xrandr si un écran est connecté en plus de l'écran principal """
        try:
            output = subprocess.check_output("xrandr | grep ' connected'", shell=True).decode()
            screens = output.strip().split('\n')
            return len(screens) > 1 # Vrai s'il y a plus d'un écran
        except:
            return False

    def image_callback(self, msg):
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            if self._generated_dir is not None:
                self._image_index += 1
                ts = datetime.datetime.now().strftime("%H-%M-%S")
                fname = f"{self._image_index:02d}_genere_{ts}.jpg"
                cv2.imwrite(os.path.join(self._generated_dir, fname), self.current_image)
                self.get_logger().info(f"Dessin généré sauvegardé : {fname}")
        except Exception as e:
            self.get_logger().error(f"Erreur décodage : {e}")

    def update_gui(self):
        if self.current_image is not None:
            cv2.imshow(self.window_name, self.current_image)
        else:
            cv2.imshow(self.window_name, np.zeros((1080, 1920, 3), dtype=np.uint8))
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = ProjectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()
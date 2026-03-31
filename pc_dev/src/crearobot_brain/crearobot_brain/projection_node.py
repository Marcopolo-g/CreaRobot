import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import subprocess

class ProjectionNode(Node):
    def __init__(self):
        super().__init__('projection_node')
        self.bridge = CvBridge()
        
        # --- CONFIGURATION ---
        self.projector_x_offset = 1920 
        self.window_name = "Projection_QT"
        self.current_image = None 


        # --- SÉCURITÉ HDMI ---
        if not self.is_projector_connected():
            self.get_logger().error("Projecteur NON DÉTECTÉ ! Mode fenêtre sur écran principal activé.")
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            # Pas de plein écran si pas de projecteur
        else:
            self.get_logger().info("Projecteur détecté. Déploiement plein écran...")
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.moveWindow(self.window_name, self.projector_x_offset, 0)
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # Subscriber
        self.subscription = self.create_subscription(Image, '/pc/projector/image', self.image_callback, 10)

        # Timer pour rafraîchir l'image (20 FPS)
        #self.gui_timer = self.create_timer(0.05, self.update_gui)

        self.get_logger().info(f"Projection initialisée sur le projecteur (Offset: {self.projector_x_offset})")

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
        except Exception as e:
            self.get_logger().error(f"Erreur décodage : {e}")

    def update_gui(self):
        if self.current_image is not None:
            cv2.imshow(self.window_name, self.current_image)
        else:
            # On affiche du noir en attendant une image
            # Si ton projecteur a une résolution différente, tu peux ajuster (1080, 1920)
            black_screen = np.zeros((1080, 1920, 3), dtype=np.uint8)
            cv2.imshow(self.window_name, black_screen)
        
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
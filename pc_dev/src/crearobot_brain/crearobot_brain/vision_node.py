import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import base64
import openai
import os
from . import config

class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')
        openai.api_key = config.OPENAI_API_KEY
        self.bridge = CvBridge()
        
        # --- CONFIGURATION CAMÉRA ---
        # 0 est l'index par défaut de la webcam PC
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error("Impossible d'ouvrir la caméra !")
        
        self.latest_frame = None

        # --- PUBLISHERS / SUBSCRIPTIONS ---
        # On publie quand même le flux pour pouvoir voir ce que le robot voit (debug)
        self.image_pub = self.create_publisher(Image, '/image_raw', 10)
        
        # On écoute le trigger
        self.trigger_sub = self.create_subscription(String, '/pc/camera/trigger', self.handle_trigger, 10)
        
        # Publisher pour le feedback de l'IA
        self.feedback_pub = self.create_publisher(String, '/pc/vision/feedback', 10)

        # --- TIMERS ---
        # Boucle de capture à 20 FPS pour garder l'image "fraîche"
        self.timer = self.create_timer(0.05, self.capture_loop)
        
        self.get_logger().info("Vision Node (OpenCV Direct) prêt !")

    def capture_loop(self):
        """ Lit la caméra en continu et publie pour le debug """
        ret, frame = self.cap.read()
        if ret:
            self.latest_frame = frame
            # On publie en ROS2 pour pouvoir utiliser rqt_image_view
            msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            self.image_pub.publish(msg)

    def handle_trigger(self, msg):
        if msg.data == "ANALYZE_C1":
            if self.latest_frame is None:
                self.get_logger().warn("Trigger reçu mais pas d'image en mémoire !")
                return
            
            self.get_logger().info("Analyse du dessin demandée...")
            self.process_with_ai()

    def process_with_ai(self):
        try:
            # Sauvegarde locale
            path = os.path.expanduser("~/Desktop/CreaRobot/last_capture.jpg")
            cv2.imwrite(path, self.latest_frame)
            
            # Encodage Base64
            _, buffer = cv2.imencode('.jpg', self.latest_frame)
            base64_img = base64.b64encode(buffer).decode('utf-8')
            
            # Prompt et Appel GPT-4o-mini
            prompt = "Tu es Hyppolite, un robot ami des enfants. Commente brièvement ce dessin et pose une question."
            
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                    ]
                }],
                max_tokens=80
            )
            
            feedback = response.choices[0].message.content
            self.get_logger().info(f"IA : {feedback}")
            
            # Publication du résultat
            res_msg = String()
            res_msg.data = feedback
            self.feedback_pub.publish(res_msg)

        except Exception as e:
            self.get_logger().error(f"Erreur Vision/IA : {e}")

    def __del__(self):
        # Libère la caméra proprement quand on arrête le node
        if self.cap.isOpened():
            self.cap.release()

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
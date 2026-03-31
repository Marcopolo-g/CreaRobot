import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import cv2
import base64
import openai
import os
from . import config

class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')
        openai.api_key = config.OPENAI_API_KEY
        self.trigger_sub = self.create_subscription(String, '/pc/camera/trigger', self.handle_trigger, 10)
        self.feedback_pub = self.create_publisher(String, '/pc/vision/feedback', 10)
        self.get_logger().info("Vision Node prêt (Attente Trigger unique).")

    def handle_trigger(self, msg):
        if msg.data == "ANALYZE_C1":
            self.get_logger().info("Trigger reçu, tentative de capture...")
            
            # On ouvre la caméra JUSTE pour la photo
            cap = cv2.VideoCapture(1) 

            if not cap.isOpened():
                self.get_logger().error("ÉCHEC : Caméra inaccessible (occupée ou débranchée).")
                return

            # On vide le buffer (important pour avoir du frais)
            for _ in range(5):
                cap.read()
            
            ret, frame = cap.read()
            cap.release()

            if ret:
                self.process_analysis(frame)
            else:
                self.get_logger().error("ÉCHEC : Capture impossible malgré l'ouverture.")

    def process_analysis(self, frame):
        try:
            # Encodage et envoi IA (ton code précédent est bon ici)
            _, buffer = cv2.imencode('.jpg', frame)
            base64_img = base64.b64encode(buffer).decode('utf-8')
            
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": [{"type": "text", "text": "Commente ce dessin."}, 
                          {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}]}],
                max_tokens=60
            )
            
            res = String()
            res.data = response.choices[0].message.content
            self.feedback_pub.publish(res)
            self.get_logger().info("Analyse envoyée.")
            
        except Exception as e:
            self.get_logger().error(f"Erreur IA : {e}")

def main():
    rclpy.init()
    node = VisionNode()
    rclpy.spin(node)
    rclpy.shutdown()
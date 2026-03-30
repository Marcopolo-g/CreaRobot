import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
import cv2
import numpy as np
from cv_bridge import CvBridge
import base64
import openai
import os
import json

from . import config

class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')
        
        # Initialisation
        self.bridge = CvBridge()
        self.latest_frame = None
        openai.api_key = config.OPENAI_API_KEY
        
        # --- SUBSCRIPTIONS ---
        # On récupère le flux d'images de la Gateway
        self.image_sub = self.create_subscription(Image, '/pc/camera/image', self.image_callback, 1)
        
        # On écoute le signal de l'InteractionNode
        self.trigger_sub = self.create_subscription(String, '/pc/camera/trigger', self.handle_trigger, 10)
        
        # --- PUBLISHERS ---
        # On renvoie le feedback textuel final
        self.feedback_pub = self.create_publisher(String, '/pc/vision/feedback', 10)
        
        self.get_logger().info("--- VISION NODE READY (Capture + Analyse C1) ---")

    def image_callback(self, msg):
        """ Stocke en continu la dernière image reçue du robot """
        self.latest_frame = msg

    def handle_trigger(self, msg):
        """ Se déclenche quand l'InteractionNode dit 'ANALYZE_C1' """
        if msg.data == "ANALYZE_C1":
            if self.latest_frame is None:
                self.get_logger().warn("Trigger reçu mais aucune image en mémoire !")
                return
            
            self.get_logger().info("Capture et Analyse en cours...")
            self.process_vision_sequence()

    def process_vision_sequence(self):
        """ Séquence complète : Save -> Encode -> API -> Publish """
        try:
            # Conversion et Sauvegarde de la photo
            # On utilise bgr8 car OpenCV écrit en BGR par défaut
            cv_image = self.bridge.imgmsg_to_cv2(self.latest_frame, desired_encoding='bgr8')
            filename = config.IMAGE_PATH
            cv2.imwrite(filename, cv_image)
            self.get_logger().info(f"Image sauvegardée sous : {filename}")

            # Encodage en Base64
            base64_image = self.encode_image(filename)
            
            # Analyse VLM
            self.run_vlm_analysis(base64_image)

        except Exception as e:
            self.get_logger().error(f"Erreur dans la séquence de vision : {e}")

    def encode_image(self, image_path):
        """ Prépare l'image pour l'API OpenAI """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def run_vlm_analysis(self, base64_image):
        """ Envoie l'image à GPT-4o avec le prompt spécifique C1 """
        
        # Prompt Hyppolite pour la Condition 1, A MODIFIER
        prompt_c1 = """
        Tu es Hyppolite, un robot social amical et curieux qui aide les enfants à dessiner.
        Regarde ce dessin réalisé par l'enfant. Analyse-le en ignorant les éléments extérieurs (mains, table).
        
        Génère une réponse courte (30-40 mots maximum) :
        1. Identifie un élément précis (une couleur vive, une forme, un personnage).
        2. Donne un compliment sincère sur la créativité.
        3. Pose une question ouverte simple pour que l'enfant t'explique son dessin.
        
        Ta réponse doit être chaleureuse et utiliser un langage adapté à un enfant.
        """

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_c1},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=150,
            )

            feedback_text = response.choices[0].message.content
            self.get_logger().info(f"Analyse réussie : {feedback_text}")

            # Publication pour l'InteractionNode
            res_msg = String()
            res_msg.data = feedback_text
            self.feedback_pub.publish(res_msg)

        except Exception as e:
            self.get_logger().error(f"Erreur API OpenAI : {e}")

def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
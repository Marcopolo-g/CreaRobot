import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import openai
from . import config

class BrainNode(Node):
    def __init__(self):
        super().__init__('brain_node')
        openai.api_key = config.OPENAI_API_KEY
        
        self.visual_memory = "" 
        self.chat_history = []
        
        # Subs
        self.img_sub = self.create_subscription(String, '/pc/vision/image_raw_b64', self.handle_first_vision, 10)
        self.stt_sub = self.create_subscription(String, '/pc/stt/transcript', self.handle_dialogue, 10)
        
        # Pub
        self.tts_pub = self.create_publisher(String, '/pc/vision/feedback', 10)
        
        self.get_logger().info("Brain Node prêt")

    def handle_first_vision(self, msg):
        """ Étape 1 : Analyse simple de l'image """
        self.get_logger().info("Analyse visuelle du dessin...")
        
        try:
            # On demande une description SIMPLE. C'est ce qui évite les refus de sécurité.
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": "Décris précisément les formes, les couleurs et les objets de ce dessin."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{msg.data}"}}
                    ]}
                ],
                max_tokens=300
            )
            
            self.visual_memory = response.choices[0].message.content
            self.get_logger().info(f"Dessin mémorisé : {self.visual_memory[:50]}...")

            # Étape 2 : Maintenant qu'on a la mémoire, on génère le PREMIER message de QT
            self.generate_first_speech()

        except Exception as e:
            self.get_logger().error(f"Erreur Vision : {e}")

    def generate_first_speech(self):
        """ Génère le 'Bonjour' en utilisant la mémoire visuelle """
        try:
            prompt = f"{config.PROMPT_C1}\n\nVoici ce que tu vois sur le dessin : {self.visual_memory}\nCommence la discussion maintenant."
            
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": prompt}],
                max_tokens=50
            )
            
            first_speech = response.choices[0].message.content
            self.send_to_robot(first_speech)
            
            if config.ENABLE_HISTORY:
                self.chat_history.append({"role": "assistant", "content": first_speech})

        except Exception as e:
            self.get_logger().error(f"Erreur Premier Speech : {e}")

    def handle_dialogue(self, msg):
        """ Dialogue fluide en texte seul """
        if not self.visual_memory: return

        user_input = msg.data
        messages = [{"role": "system", "content": config.PROMPT_C1}]
        messages.append({"role": "system", "content": f"Mémoire du dessin : {self.visual_memory}"})
        
        if config.ENABLE_HISTORY:
            messages.extend(self.chat_history[-config.HISTORY_LIMIT:])
            self.chat_history.append({"role": "user", "content": user_input})
        
        messages.append({"role": "user", "content": user_input})

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=60
            )
            
            answer = response.choices[0].message.content
            if config.ENABLE_HISTORY:
                self.chat_history.append({"role": "assistant", "content": answer})
            
            self.send_to_robot(answer)
            
        except Exception as e:
            self.get_logger().error(f"Erreur Dialogue : {e}")

    def send_to_robot(self, text):
        msg = String()
        msg.data = text
        self.tts_pub.publish(msg)

def main():
    rclpy.init()
    node = BrainNode()
    rclpy.spin(node)
    rclpy.shutdown()
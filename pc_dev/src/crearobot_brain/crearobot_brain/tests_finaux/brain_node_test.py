import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import openai
import socket
import json
import time
import threading

from .. import config


class BrainNode(Node):
    def __init__(self):
        super().__init__('brain_node_test')
        
        openai.api_key = config.OPENAI_API_KEY

        if not self.is_internet_available():
            self.get_logger().error("PAS D'INTERNET : Le cerveau ne pourra pas contacter OpenAI.")
        else:
            self.get_logger().info("Internet OK.")

        self.history = []
        self.publisher_ = self.create_publisher(String, '/pc/qtaction', 10)

        self.warm_up_brain()
        self.get_logger().info(f"Cerveau prêt ! Mode STT : {config.STT_MODE}")

        self.terminal_thread = threading.Thread(
            target=self.terminal_callback, daemon=True)
        self.terminal_thread.start()

        self.subscription = self.create_subscription(String, '/pc/user_speech', self.speech_callback, 10)

    def is_internet_available(self):
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except OSError:
            return False

    def warm_up_brain(self):
        try:
            openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "."}],
                max_tokens=1
            )
            self.get_logger().info("Cerveau chaud et prêt !")
        except Exception:
            self.get_logger().warn("Échec du préchauffage, la première réponse sera lente.")

    # ENTRÉE 1 : MICRO
    def speech_callback(self, msg):
        self.t_received = time.time()  # heure d'arrivée du texte
        self.get_logger().info(f"Texte reçu du robot : {msg.data}")
        self.process_with_llm(msg.data)

    # ENTRÉE 2 : TERMINAL
    def terminal_callback(self):
        while rclpy.ok():
            try:
                user_text = input("[Terminal] Tape ton message : \n")
                if user_text.strip():
                    self.process_with_llm(user_text)
            except EOFError:
                break

    # LOGIQUE LLM UNIFIÉE
    def process_with_llm(self, user_input):
        self.t_received = time.time()
        consigne = f"""
        Tu es un assistant intégré dans un robot social.
        Tes réponses doivent impérativement être une liste Python au format : ["geste", "emotion", "texte"].
        
        RÈGLES STRICTES :
        1. Choisis le geste uniquement parmi : [{config.LISTE_GESTURES}].
        2. Choisis l'émotion uniquement parmi : [{config.LISTE_EMOTIONS}].
        3. Le texte doit être en français.
        """

        messages_a_envoyer = [{"role": "system", "content": consigne}]

        if config.ENABLE_HISTORY:
            contexte = self.history[-(config.HISTORY_LIMIT * 2):]
            messages_a_envoyer.extend(contexte)

        messages_a_envoyer.append({"role": "user", "content": user_input})

        # Paramètres selon le mode
        use_stream = (config.STT_MODE == "whisper")

        try:
            t_llm_start = time.time()

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages_a_envoyer,
                stream=use_stream,
                max_tokens=80,
                temperature=0.7
            )

            if use_stream:
                reponse_ia = ""
                for chunk in response:
                    reponse_ia += chunk['choices'][0]['delta'].get('content', '')
            else:
                reponse_ia = response.choices[0].message.content

            t_llm_end = time.time()

            # Publication
            msg = String()
            msg.data = reponse_ia
            self.publisher_.publish(msg)

            t_published = time.time()

            # RAPPORT COMPLET
            t_llm       = t_llm_end - t_llm_start
            t_total     = t_published - self.t_received
            self.get_logger().info("=" * 40)
            self.get_logger().info(f"LLM ({('STREAM' if use_stream else 'BATCH')}) : {t_llm:.2f}s")
            self.get_logger().info(f"Total (réception → publish) : {t_total:.2f}s")
            self.get_logger().info(f"Réponse : {reponse_ia}")
            self.get_logger().info("=" * 40)

            if config.ENABLE_HISTORY:
                self.history.append({"role": "user",     "content": user_input})
                self.history.append({"role": "assistant", "content": reponse_ia})

        except Exception as e:
            self.get_logger().error(f"Erreur LLM : {e}")


def main(args=None):
    rclpy.init(args=args)
    node = BrainNode()
    rclpy.spin(node)
    rclpy.shutdown()
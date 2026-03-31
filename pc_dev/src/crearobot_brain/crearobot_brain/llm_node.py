import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import openai
import json
import time
import threading
from . import config

class LLMReactifNode(Node):
    def __init__(self):
        super().__init__('llm_reactif_node')
        
        # Configuration OpenAI
        openai.api_key = config.OPENAI_API_KEY
        self.history = []
        self.is_active = False # Par défaut, le nœud attend l'ordre d'activation
        
        # Publisher pour les actions de QT (Gestes + Paroles)
        self.action_pub = self.create_publisher(String, '/pc/qtaction', 10)
        
        # Subscriber aux ordres de l'InteractionNode
        self.ctrl_sub = self.create_subscription(
            String, '/pc/llm_control', self.control_callback, 10)
        
        # Subscriber pour la voix de l'utilisateur (Micro)
        self.speech_sub = self.create_subscription(
            String, '/pc/user_speech', self.speech_callback, 10)

        self.get_logger().info("Nœud LLM Réactif en ligne. En attente du Dispatcher...")

        # Thread terminal pour les tests manuels
        self.terminal_thread = threading.Thread(target=self.terminal_loop, daemon=True)
        self.terminal_thread.start()

    def control_callback(self, msg):
        """ Reçoit les ordres ACTIVATE_REACTIVE ou DEACTIVATE_REACTIVE de l'InteractionNode """
        cmd = msg.data
        if cmd == "ACTIVATE_REACTIVE":
            self.is_active = True
            self.get_logger().info("Mode REACTIF activé.")
        elif cmd == "DEACTIVATE_REACTIVE":
            self.is_active = False
            self.get_logger().info("Mode REACTIF mis en veille.")

    def speech_callback(self, msg):
        """ Traite la voix seulement si le dispatcher l'a autorisé """
        if not self.is_active:
            return
            
        user_text = msg.data
        self.get_logger().info(f"Analyse réactive : {user_text}")
        self.process_with_llm(user_text)

    def terminal_loop(self):
        """ Permet de tester dans le terminal sans parler au robot """
        while rclpy.ok():
            try:
                user_text = input()
                if user_text.strip():
                    if self.is_active:
                        self.process_with_llm(user_text)
                    else:
                        print("[Warning] Le nœud LLM est désactivé par le dispatcher.")
            except EOFError:
                break

    def process_with_llm(self, user_input):
        # Prompt de base pour Hyppolite ATTTTTENNNTIONNNN CHANGER LE PROMPT CLAUDIE
        system_prompt = f"""
        Tu es Hyppolite, un robot social assistant.
        Tu réponds de manière courte et encourageante pendant que l'étudiant dessine.
        
        Format de sortie impératif (JSON list) : ["geste", "emotion", "texte"]
        Gestes autorisés : {config.LISTE_GESTURES}
        Émotions autorisées : {config.LISTE_EMOTIONS}
        """

        messages = [{"role": "system", "content": system_prompt}]
        
        if config.ENABLE_HISTORY:
            # On prend les derniers échanges pour garder le fil de la discussion
            messages.extend(self.history[-(config.HISTORY_LIMIT * 2):])
        
        messages.append({"role": "user", "content": user_input})

        try:
            start_t = time.time()
            # Note : Tu utilises peut-être la librairie openai >= 1.0.0, 
            # j'ai gardé la syntaxe 0.28 pour correspondre à ton code précédent.
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=80,
                temperature=0.7
            )
            
            reponse_ia = response.choices[0].message.content
            
            # On publie la réponse JSON vers la Gateway
            msg = String()
            msg.data = reponse_ia
            self.action_pub.publish(msg)

            # Mise à jour de l'historique
            if config.ENABLE_HISTORY:
                self.history.append({"role": "user", "content": user_input})
                self.history.append({"role": "assistant", "content": reponse_ia})

            self.get_logger().info(f"Hyppolite a répondu en {time.time()-start_t:.2f}s")

        except Exception as e:
            self.get_logger().error(f"Erreur lors de l'appel OpenAI : {e}")

def main(args=None):
    rclpy.init(args=args)
    node = LLMReactifNode()
    rclpy.spin(node)
    rclpy.shutdown()
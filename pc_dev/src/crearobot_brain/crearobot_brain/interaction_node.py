import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json

class InteractionNode(Node):
    def __init__(self):
        super().__init__('interaction_node')
        
        # Écoute les ordres de l'orchestrateur
        self.sub = self.create_subscription(String, '/pc/phase_control', self.execute_phase, 10)
        
        # Parle à la Gateway (pour les gestes et la voix directe)
        self.action_pub = self.create_publisher(String, '/pc/qtaction', 10)
        
        # --- DÉLÉGATION ---
        # Pour activer/désactiver le LLM Réactif
        self.llm_ctrl_pub = self.create_publisher(String, '/pc/llm_control', 10)
        # Pour déclencher la caméra
        self.camera_trigger_pub = self.create_publisher(String, '/pc/camera/trigger', 10)

    def execute_phase(self, msg):
        cmd = msg.data
        self.get_logger().info(f"Réception commande : {cmd}")

        if cmd == "START_PHASE_1":
            self.send_robot("hi", "happy", "Bonjour ! Je suis Hyppolite. On va dessiner ensemble. Es-tu prêt ?")
            
        elif cmd == "START_PHASE_2":
            # Le robot fait son intro
            self.send_robot("challenge", "happy", "C'est parti ! Complète le dessin sur la feuille.")
            # On réveille le noeud LLM Réactif
            self.llm_ctrl_pub.publish(String(data="ACTIVATE_REACTIVE"))
            
        elif cmd == "START_PHASE_3":
            # On endort le LLM Réactif (pour qu'il ne réponde pas pendant l'analyse)
            self.llm_ctrl_pub.publish(String(data="DEACTIVATE_REACTIVE"))
            # On demande la photo
            self.camera_trigger_pub.publish(String(data="TAKE_PHOTO"))
            # Le robot réagit
            self.send_robot("head_scratch", "curious", "Laisse-moi regarder ton dessin... C'est intéressant.")
            # On pourra appeler le noeud LLM Feedback ici plus tard
        
        elif cmd == "START_PHASE_4":
            self.send_robot("challenge", "happy", "C'est parti pour la phase 4 ! Tu peux continuer.")
            self.llm_ctrl_pub.publish(String(data="ACTIVATE_REACTIVE"))

        elif cmd == "START_PHASE_5":
            self.llm_ctrl_pub.publish(String(data="DEACTIVATE_REACTIVE"))
            self.send_robot("bye", "happy", "On a bien travaillé ! À bientôt !")

    def send_robot(self, geste, emotion, texte):
        msg = String()
        msg.data = json.dumps([geste, emotion, texte])
        self.action_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = InteractionNode()
    rclpy.spin(node)
    rclpy.shutdown()
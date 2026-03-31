import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
from . import config

class InteractionNode(Node):
    def __init__(self):
        super().__init__('interaction_node')
        
        # Écoute les ordres de l'orchestrateur
        self.sub = self.create_subscription(String, '/pc/phase_control', self.execute_phase, 10)
        
        # Parle à la Gateway (pour les gestes et la voix directe)
        self.action_pub = self.create_publisher(String, '/pc/qtaction', 10)
        
        # --- DÉLÉGATION ---
        # Pour activer/désactiver le LLM Réactif
        # JE NE VAIS PAS LUTILISER AU FINAL
        # self.llm_ctrl_pub = self.create_publisher(String, '/pc/llm_control', 10)
        
        # Pour déclencher la caméra
        self.camera_trigger_pub = self.create_publisher(String, '/pc/camera/trigger', 10)
        
        # Pour le feedback ecrit de C1
        self.vision_feedback_sub = self.create_subscription(String, '/pc/vision/feedback', self.receive_vlm_feedback, 10)

    def execute_phase(self, msg):
        cmd = msg.data
        self.get_logger().info(f"Réception commande : {cmd}")

        if cmd == "START_PHASE_1":
            self.send_robot("hi", "happy", "Bonjour ! On va dessiner ensemble. Es-tu prêt ?")
            
        elif cmd == "START_PHASE_2":
            # Le robot fait son intro
            self.send_robot("challenge", "happy", "C'est parti ! Complète le dessin sur la feuille.")
            
        elif cmd == "START_PHASE_3":
            # On lance la logique spécifique à la condition (C0, C1 ou C2)
            self.start_phase_3()
        
        elif cmd == "START_PHASE_4":
            self.send_robot("challenge", "happy", "C'est parti pour la phase 4 ! Tu peux continuer.")

        elif cmd == "START_PHASE_5":
            self.send_robot("bye", "happy", "On a bien travaillé ! À bientôt !")

    def send_robot(self, geste, emotion, texte):
        msg = String()
        msg.data = json.dumps([geste, emotion, texte])
        self.action_pub.publish(msg)

    def start_phase_3(self):
        if config.CONDITION == "C0":
            # Envoi direct du texte pré-enregistré
            self.send_robot("happy", "happy", config.FEEDBACK_C0)

        elif config.CONDITION == "C1":
            self.send_robot("curious", "neutral", "Laisse-moi regarder ton dessin quelques instants...")
            # On demande au VisionNode de travailler
            trigger_msg = String()
            trigger_msg.data = "ANALYZE_C1"
            self.camera_trigger_pub.publish(trigger_msg)


        elif config.CONDITION == "C2":
            # Petite phrase d'introduction
            intro = "Oh, j’ai regardé votre dessin et j’ai eu envie d’essayer quelque chose !"
            self.send_robot("surprise", "happy", intro)
            # Génération de la complétion (VLM + Image Gen)
            completion_url = self.vision_module.generate_completion(config.IMAGE_PATH)
            # Affichage/Projection de l'image (via un topic ROS2 dédié)
            self.display_image(completion_url)
            # Phrase de conclusion
            self.send_robot("hi", "happy", "Maintenant, à vous de continuer !")

    # Callback pour C1
    def receive_vlm_feedback(self, msg):
        # Dès que le VisionNode a fini, on fait parler le robot
        self.send_robot("happy", "happy", msg.data)

def main(args=None):
    rclpy.init(args=args)
    node = InteractionNode()
    rclpy.spin(node)
    rclpy.shutdown()
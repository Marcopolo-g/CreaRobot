# interaction_node.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Int32, Float64MultiArray
import json
from . import config

class InteractionNode(Node):
    def __init__(self):
        super().__init__('interaction_node')
        

        # Écoute les ordres de phases de l'orchestrateur
        self.sub = self.create_subscription(String, '/pc/phase_control', self.execute_phase, 10)

        # Ecoute le numero de la loop de l'orchestrateur
        self.tour_sub = self.create_subscription(Int32, 'pc/tour_control', self.tour_callback, 10)

        # Retour du Brain (Feedback LLM/VLM)
        self.vision_feedback_sub = self.create_subscription(String, '/pc/vision/feedback', self.receive_vlm_feedback, 10)
        
        # Écoute du transcript pour verrouiller l'interaction
        self.stt_sub = self.create_subscription(String, '/pc/stt/transcript', self.stt_callback, 10)
        

        # Parle à la Gateway (pour les gestes et la voix directe)
        self.action_pub = self.create_publisher(String, '/pc/qtaction', 10)

        # Pour trigger la camera 
        self.camera_trigger_pub = self.create_publisher(String, '/pc/camera/trigger', 10)
        
        # Contrôle du STT (Activation micro)
        self.stt_enable_pub = self.create_publisher(Bool, '/pc/stt/enable', 10)
        
        # On previent l'orchestrateur quand llm a fini son intercation (durant le feedback)
        self.loop_done_pub = self.create_publisher(String, '/pc/interaction_finished', 10)

        # Publisher pour commander la tête (Yaw, Pitch)
        self.head_pub = self.create_publisher(Float64MultiArray, '/pc/head_position/command', 10)

        self.is_busy = False
        self.current_cmd = ""
        self.current_tour = 1 
        self.feedback_timer = None
        self.dialogue_count = 0
        self.photo_timer = None


    def set_stt(self, state):
        msg = Bool()
        msg.data = state
        self.stt_enable_pub.publish(msg)

    def execute_phase(self, msg):
        self.current_cmd = msg.data
        self.is_busy = False

        if self.feedback_timer:
            self.feedback_timer.cancel()
            self.feedback_timer = None
        
        if "START_INTRO" in self.current_cmd:
            self.move_head(0, 0)
            text = "Bonjour ! Je m'appelle QT. Je suis ravi de faire ta connaissance. Es-tu prêt pour notre activité ?"
            self.send_robot("hi", "happy", text)
            # On ouvre le micro si besoin pour le "pret/oui"

            duree = self.calculate_speech_duration(text) + 4
            self.feedback_timer = self.create_timer(duree, self.finish_feedback_loop)
            
        elif "START_ICE_BREAKING" in self.current_cmd:  
            self.move_head(0, 0)
            self.dialogue_count = 0

            text = "Super ! Avant de commencer, dis-moi, est-ce que tu aimes dessiner d'habitude ?"
            self.send_robot("happy", "happy", text)

            self.set_stt(True)
            
            duree = self.calculate_speech_duration(text) 
            self.feedback_timer = self.create_timer(duree, self.finish_feedback_loop)
            

        elif "START_TASK_INTRO" in self.current_cmd:
            self.move_head(0, 0)
            text = "Je vais te donner une feuille avec des petites formes dessus — et ton objectif, c'est de les utiliser pour faire un dessin. Tu peux dessiner ce que tu veux, comme tu veux. Il n'y a pas de bonne ou de mauvaise réponse. Si tu as fini le dessin avant la fin du temps, tu peux dire que tu as fini. Es-tu prêt ?"
            self.send_robot("talk", "neutral", text)
            # C'est un monologue, on envoie DONE à la fin du temps de parole
            duree = self.calculate_speech_duration(text) + 3
            self.feedback_timer = self.create_timer(duree, self.finish_feedback_loop)

        elif "START_DRAWING" in self.current_cmd:
            self.move_head(0, 20)

            # ----------- A MODIFIER -------------------------------------------------
            if self.current_tour == 1:
                self.send_robot("curious", "neutral", "C’est parti !")
            elif self.current_tour == 2:
                self.send_robot("challenge", "neutral", "C’est parti pour la suite !")
            elif self.current_tour == 3:
                self.send_robot("curious", "neutral_state_blinking", "Continue !")
            self.set_stt(True) # On ouvre pour detecter le "j'ai fini"
            
        elif "START_FEEDBACK" in self.current_cmd:
            self.move_head(0, 0)
            self.dialogue_count = 0 # reset du compteur pour chaque phase de feedback
            self.trigger_feedback()
        
        elif "START_TITLE" in self.current_cmd:
            # Le robot se penche pour admirer l'œuvre finale
            self.move_head(0, 20)
            text = "C'est fini ! Laisse-moi admirer ton œuvre une dernière fois pour lui trouver un titre..."
            self.send_robot("happy", "happy", text)
            
            self.set_stt(False)
            # On déclenche la photo après la phrase
            wait_time = self.calculate_speech_duration(text) + 2
            if self.photo_timer: self.photo_timer.cancel()
            self.photo_timer = self.create_timer(wait_time, self.send_camera_trigger)
            
        elif "START_ENDING" in self.current_cmd:
            self.move_head(0, 0)
            self.set_stt(False)
            self.feedback_timer = self.create_timer(1.0, self.say_goodbye_final)

    def trigger_feedback(self):
        """ Logique des conditions C0, C1, C2 """
        text = ""

        if config.CONDITION == "C0":
            if self.current_tour == 1:
                text = config.FEEDBACK_C0_1
                self.send_robot("happy", "happy", text)
                
            elif self.current_tour == 2:
                text = config.FEEDBACK_C0_2
                self.send_robot("surprise", "surprise", text)
            
            else:
                return

            # On calcule dynamiquement le temps de parole
            if self.feedback_timer: self.feedback_timer.cancel()
            duree = self.calculate_speech_duration(text)
            self.create_timer(duree, self.finish_feedback_loop)
            
        elif config.CONDITION == "C1":
            if self.current_tour == 1:
                self.send_robot("happy", "happy", "Oh, laisse-moi regarder ton dessin… Recule un petit peu pour que la caméra puisse bien voir ton dessin.")
            elif self.current_tour == 2:
                self.send_robot("surprise", "surprise", "Je peux voir où tu en es ? Recule un petit peu pour que la caméra puisse bien voir ton dessin.")

            self.set_stt(False) # On coupe pendant l'analyse vision

            wait_time = self.calculate_speech_duration(text) + 4

            if self.photo_timer:
                self.photo_timer.cancel()
            self.photo_timer = self.create_timer(wait_time, self.send_camera_trigger)
            self.get_logger().info(f"Photo programmée dans {wait_time:.2f}s (Phrase: {len(text)} car.)")
            
        
        elif config.CONDITION == "C2":
            pass

    def send_camera_trigger(self):
        """ Callback du timer pour envoyer le trigger caméra """
        if self.photo_timer:
            self.photo_timer.cancel()
            self.photo_timer = None
        
        # C'est ici qu'on envoie réellement l'ordre à la caméra
        trigger_msg = String(data="ANALYZE_IMAGE")
        self.camera_trigger_pub.publish(trigger_msg)
        self.get_logger().info("Déclenchement photo envoyé !")

    def receive_vlm_feedback(self, msg):
        """ Reponse du LLM/VLM recue """
        self.is_busy = True
        self.set_stt(False)

        if self.feedback_timer:
            self.feedback_timer.cancel()
        
        self.send_robot("None", "None", msg.data)
        
        self.dialogue_count += 1

        # Calcul du temps de parole pour deverrouiller
        duree = self.calculate_speech_duration(msg.data)
        self.feedback_timer = self.create_timer(duree, self.finish_feedback_loop) 
           
    def calculate_speech_duration(self, text):
        # Estimation : 0.08s par caractere + une marge de securite
        return (len(text) * 0.08) + 1

    def stt_callback(self, msg):
        if self.is_busy:
            return
        
        if "START_FEEDBACK" in self.current_cmd:
            self.is_busy = True
            self.set_stt(False)
            self.get_logger().info(f"Dialogue Feedback verrouillé pour : {msg.data}")
        else:
            # En INTRO ou DRAWING, on se contente de logger, sans rien couper.
            # L'Orchestrateur recevra aussi le message et fera son travail de son côté.
            self.get_logger().info(f"Audio capté (Phase {self.current_cmd}) : {msg.data}")


    def finish_feedback_loop(self):
        """ Libere le micro et previent l'orchestrateur pour la suite """
        if self.feedback_timer:
            self.feedback_timer.cancel()
            self.feedback_timer = None

        self.is_busy = False

        if "START_INTRO" in self.current_cmd:
            self.loop_done_pub.publish(String(data="DONE"))

        # --- LOGIQUE DE SORTIE POUR ICE_BREAKING ET FEEDBACK ---
        elif "START_ICE_BREAKING" in self.current_cmd:
            if self.dialogue_count < config.DIALOGUE_DURATION:
                self.set_stt(True)
                self.get_logger().info("Ice Breaking : On continue la discussion.")
            else:
                self.get_logger().info("Ice Breaking fini. Envoi de DONE.")
                self.loop_done_pub.publish(String(data="DONE"))

        elif "START_TASK_INTRO" in self.current_cmd:
            self.loop_done_pub.publish(String(data="DONE"))

        
        # Sortie du Feedback classique (C0 ou C1)
        elif "START_FEEDBACK" in self.current_cmd:
            if config.CONDITION == "C0":
                self.loop_done_pub.publish(String(data="DONE"))
            elif config.CONDITION == "C1":
                if self.dialogue_count < config.MAX_EXCHANGES:
                    self.set_stt(True)
                else:
                    self.loop_done_pub.publish(String(data="DONE"))
            
        elif "START_TITLE" in self.current_cmd:
            self.loop_done_pub.publish(String(data="DONE"))


    def say_goodbye_final(self):
        """ Callback pour déclencher la parole après le mouvement de tête """
        if self.feedback_timer:
            self.feedback_timer.cancel()
            self.feedback_timer = None
            
        self.send_robot("bye", "happy", "Voilà, le temps est écoulé ! Merci d'avoir dessiné avec moi aujourd'hui !")

    def tour_callback(self, msg):
        """Met à jour le numéro du tour actuel envoyé par l'orchestrateur"""
        self.current_tour = msg.data

    def send_robot(self, geste, emotion, texte):
        msg = String()
        msg.data = json.dumps([geste, emotion, texte])
        self.action_pub.publish(msg)

    def move_head(self, yaw, pitch):
        """ Envoie une position à la tête [Yaw, Pitch] """
        msg = Float64MultiArray()
        # HeadYaw: data[0], HeadPitch: data[1]
        msg.data = [float(yaw), float(pitch)]
        self.head_pub.publish(msg)
        self.get_logger().info(f"Mouvement tête : Yaw={yaw}, Pitch={pitch}")

def main(args=None):
    rclpy.init(args=args)
    node = InteractionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
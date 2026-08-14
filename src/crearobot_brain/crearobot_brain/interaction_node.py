# interaction_node.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Int32, Float64MultiArray
from rclpy.qos import QoSProfile, DurabilityPolicy
import json
import random
import os

from . import config

_LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)

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

        # Écoute le signal réel de parole du robot (publié par gateway_node),
        # utilisé pour enchaîner les actions au bon moment plutôt que d'estimer une durée
        self.speaking_sub = self.create_subscription(Bool, '/pc/robot/speaking', self.on_robot_speaking, 10)

        # Parle à la Gateway (pour les gestes et la voix directe)
        self.action_pub = self.create_publisher(String, '/pc/qtaction', 10)

        # Pour trigger la camera 
        self.camera_trigger_pub = self.create_publisher(String, '/pc/camera/trigger', 10)
        
        # Contrôle du STT (Activation micro)
        self.stt_enable_pub = self.create_publisher(Bool, '/pc/stt/enable', _LATCHED)
        
        # On previent l'orchestrateur quand llm a fini son intercation (durant le feedback)
        self.loop_done_pub = self.create_publisher(String, '/pc/interaction_finished', 10)

        # Publisher pour commander la tête (Yaw, Pitch)
        self.head_pub = self.create_publisher(Float64MultiArray, '/pc/head_position/command', 10)

        self.is_busy = False
        self.current_cmd = ""
        self.current_tour = 1
        self.feedback_timer = None
        self.silence_timer = None
        self.photo_timer = None
        self.animation_timer = None
        self.dialogue_count = 0

        # État pour l'attente de fin de parole réelle du robot
        self._robot_is_speaking = False
        self._awaiting_speech_start = False
        self._pending_speech_state = None

    def set_stt(self, state):
        msg = Bool()
        msg.data = state
        self.stt_enable_pub.publish(msg)
        self.get_logger().info(f"--- COMMANDE STT : {'OUVERTURE' if state else 'FERMETURE'} ---")
        # Si on ouvre le micro, on peut lancer le timer de silence
        if state:
            self.reset_silence_timer()
        else:
            self.stop_silence_timer()

    def _cancel_pending_speech_action(self):
        """ Invalide toute attente de fin de parole en cours (ex. changement de phase) """
        state = self._pending_speech_state
        if state is not None:
            state['resolved'] = True
            if state.get('delay_timer'):
                state['delay_timer'].cancel()
        self._pending_speech_state = None

    def _after_speech(self, callback, extra_delay=0.0, timer_attr='feedback_timer'):
        """
        Programme callback pour qu'il s'exécute une fois que le robot a réellement fini de
        parler (transition True -> False de /pc/robot/speaking, publié par gateway_node),
        plus extra_delay secondes.
        """
        self._cancel_pending_speech_action()

        state = {'resolved': False, 'extra_delay': extra_delay, 'timer_attr': timer_attr,
                  'delay_timer': None}

        def resolve():
            if state['resolved']:
                return
            state['resolved'] = True
            if state['delay_timer']:
                state['delay_timer'].cancel()
            if self._pending_speech_state is state:
                self._pending_speech_state = None
            callback()

        state['resolve'] = resolve

        self._pending_speech_state = state
        self._awaiting_speech_start = not self._robot_is_speaking

    def on_robot_speaking(self, msg):
        self._robot_is_speaking = bool(msg.data)

        if self._robot_is_speaking:
            # Le robot a commencé à parler : la prochaine transition à False sera la bonne
            self._awaiting_speech_start = False
            return

        if self._pending_speech_state is None or self._awaiting_speech_start:
            # Pas d'attente en cours, ou ce False appartient à une parole précédente
            return

        state = self._pending_speech_state

        if state['extra_delay'] > 0:
            timer = self.create_timer(state['extra_delay'], state['resolve'])
            state['delay_timer'] = timer
            setattr(self, state['timer_attr'], timer)
        else:
            state['resolve']()

    def execute_phase(self, msg):
        data = json.loads(msg.data)
        self.current_cmd  = data["phase"]
        self.current_tour = data["tour"]
        self._triggered_by = data.get("triggered_by", "")
        self.set_stt(False)
        self.is_busy = False

        self.stop_random_animation()
        self.stop_silence_timer()
        self._cancel_pending_speech_action()

        if self.feedback_timer:
            self.feedback_timer.cancel()
            self.feedback_timer = None

        if "START_INTRO" in self.current_cmd:
            self.move_head(0, 0)
            text = "Bonjour ! #LAUGH01# Je m'appelle Q.T.. Je suis ravi de faire ta connaissance. Es-tu prêt pour notre activité ?"
            self.send_robot("hi", "happy", text)
            # On ouvre le micro si besoin pour le "pret/oui"
            self._after_speech(self.finish_feedback_loop, extra_delay=0.3)

        elif "START_ICE_BREAKING" in self.current_cmd:
            self.move_head(0, 0)
            self.dialogue_count = 0

            text = config.ICE_BREAKING_QUESTION
            self.send_robot("happy", "happy", text)
            self._after_speech(self.finish_feedback_loop, extra_delay=0.3)

        elif "START_TASK_INTRO" in self.current_cmd:
            self.move_head(0, 0)
            text = "Enfin, assez avec mes questions, revenons à notre activité. Je vais te donner une feuille avec des petites formes dessus. Ton objectif, c'est de les utiliser pour faire un dessin. Il n'y a pas de bonne ou de mauvaise réponse. Si tu as fini avant la fin du temps, tu peux me le dire."
            self.send_robot("None", "None", text)
            # C'est un monologue, on envoie DONE à la fin du temps de parole
            self._after_speech(self.finish_feedback_loop, extra_delay=5.5)

        elif "START_DRAWING" in self.current_cmd:
            self.move_head(0, config.HEAD_PITCH_DRAWING)

            # ----------- A MODIFIER -------------------------------------------------
            if self.current_tour == 1:
                self.send_robot("happy", "None", "C’est parti pour le dessin !")
            elif self.current_tour == 2:
                self.send_robot("challenge", "None", "Reprenons le dessin !")
            elif self.current_tour == 3:
                self.send_robot("challenge", "None", "Dernière modification de ton dessin !")
            #elif self.current_tour == 4:
            #    self.send_robot("curious", "neutral_state_blinking", "Continue !")
            self.set_stt(True) # On ouvre pour detecter le "j'ai fini"

            # Pour generer des animations durant la phase de dessin
            self.schedule_random_animation()
            
        elif "START_FEEDBACK" in self.current_cmd:
            self.move_head(0, 0)
            self.dialogue_count = 0
            self.trigger_feedback(skip_intro=(self._triggered_by == "addressing"))
        
        elif "START_TITLE" in self.current_cmd:
            # Le robot se penche pour admirer l'œuvre finale
            self.move_head(0, 20)
            text = "C'est fini ! #MMM01# Laisse-moi admirer ton œuvre une dernière fois pour lui trouver un titre..."
            self.send_robot("happy", "happy", text)

            self.set_stt(False)
            # On déclenche la photo après la phrase
            self._after_speech(self.send_camera_trigger, extra_delay=2.5, timer_attr='photo_timer')
            
        elif "START_ENDING" in self.current_cmd:
            self.move_head(0, 0)
            self.set_stt(False)
            self.feedback_timer = self.create_timer(1.0, self.say_goodbye_final)

    def trigger_feedback(self, skip_intro=False):
        """ Logique des conditions C0, C1, C2 """
        text = ""

        if config.CONDITION == "C0":
            if self.current_tour == 1:
                text = random.choice(config.FEEDBACK_C0_1)
                self.send_robot("happy", "happy", text)
            elif self.current_tour == 2:
                text = random.choice(config.FEEDBACK_C0_2)
                self.send_robot("surprise", "surprise", text)
            else:
                return

            self._after_speech(self.finish_feedback_loop)

        elif config.CONDITION in ("C1", "C2"):
            self.set_stt(False)

            if skip_intro:
                # Pas de phrase envoyée ici (déjà répondu à l'aveugle sur adressage) :
                # simple pause fixe avant la photo, indépendante de toute parole
                if self.photo_timer:
                    self.photo_timer.cancel()
                self.photo_timer = self.create_timer(3.5, self.send_camera_trigger)
                self.get_logger().info(f"[{config.CONDITION}] Photo programmée dans 3.5s (skip intro)")
            else:
                _phrases = {
                    1: ("happy",    "Oh, Faisons une pause dans le dessin que je puisse regarder. Recule un petit peu pour que la caméra puisse bien le voir."),
                    2: ("surprise", "Refaisons une pause. Je peux voir où tu en es ? Recule un petit peu pour que la caméra puisse bien voir ton dessin."),
                    3: ("happy",    "J'aimerais beaucoup voir tes dernières touches ! Est-ce que tu peux montrer ton dessin à la caméra une dernière fois pour que je puisse l'admirer ?"),
                }
                entry = _phrases.get(self.current_tour)
                if entry:
                    emotion, text = entry
                    self.send_robot(emotion, emotion, text)
                self._after_speech(self.send_camera_trigger, extra_delay=1.0, timer_attr='photo_timer')
                self.get_logger().info(f"[{config.CONDITION}] Photo programmée après la fin de la phrase du robot")

    def send_camera_trigger(self):
        """ Callback du timer pour envoyer le trigger caméra """
        if self.photo_timer:
            self.photo_timer.cancel()
            self.photo_timer = None
        
        # C'est ici qu'on envoie réellement l'ordre à la caméra
        trigger_msg = String(data="ANALYZE_IMAGE")
        self.camera_trigger_pub.publish(trigger_msg)
        self.get_logger().info("Déclenchement photo envoyé !")

        # Animations de réflexion pendant la génération (C2 feedback uniquement, pas le titre)
        if config.CONDITION == "C2" and "START_FEEDBACK" in self.current_cmd:
            emotions = ["bored", "neutral", "bored"]
            phrases = random.sample(config.C2_PHRASES_THINKING, k=min(3, len(config.C2_PHRASES_THINKING)))
            while len(phrases) < 3:
                phrases.append(random.choice(config.C2_PHRASES_THINKING))

            def do_thinking(i=0):
                if self.animation_timer:
                    self.animation_timer.cancel()
                    self.animation_timer = None
                self.send_robot(emotions[i], "None", phrases[i])
                if i + 1 < len(phrases):
                    self.animation_timer = self.create_timer(15.0, lambda: do_thinking(i + 1))

            self.animation_timer = self.create_timer(5.0, do_thinking)

    def receive_vlm_feedback(self, msg):
        """ Reponse du LLM/VLM recue """
        self.is_busy = True
        self.set_stt(False)

        if self.animation_timer:
            self.animation_timer.cancel()
            self.animation_timer = None

        if self.feedback_timer:
            self.feedback_timer.cancel()
        
        self.send_robot("None", "None", msg.data)

        self.dialogue_count += 1

        self._after_speech(self.finish_feedback_loop)

    def stt_callback(self, msg):
        if self.is_busy:
            return

        self.stop_silence_timer()

        if "START_FEEDBACK" in self.current_cmd or "START_ICE_BREAKING" in self.current_cmd:
            self.is_busy = True
            self.set_stt(False)
            self.get_logger().info(f"Dialogue Feedback verrouillé pour : {msg.data}")

        elif "START_DRAWING" in self.current_cmd:
            self.get_logger().info(f"Audio capté (Phase DRAWING) : {msg.data}")

        else:
            self.get_logger().info(f"Audio capté (Phase {self.current_cmd}) : {msg.data}")


    def finish_feedback_loop(self):
        """ Libere le micro et previent l'orchestrateur pour la suite """
        if self.feedback_timer:
            self.feedback_timer.cancel()
            self.feedback_timer = None

        self.is_busy = False

        if "START_INTRO" in self.current_cmd:
            self.set_stt(True)  # attend le "oui/prêt" oral

        # --- LOGIQUE DE SORTIE POUR ICE_BREAKING ET FEEDBACK ---
        elif "START_ICE_BREAKING" in self.current_cmd:
            if self.dialogue_count < config.DIALOGUE_DURATION:
                self.set_stt(True)
            else:
                self.stop_silence_timer()
                self.set_stt(False)
                self.loop_done_pub.publish(String(data=f"DONE:{self.current_cmd}"))

        elif "START_TASK_INTRO" in self.current_cmd:
            self.loop_done_pub.publish(String(data=f"DONE:{self.current_cmd}"))
            self.set_stt(False)

        elif "START_DRAWING" in self.current_cmd:
            # On ouvre pour détecter si l'utilisateur dit "J'ai fini"
            self.set_stt(True)
        
        # Sortie du Feedback classique (C0 ou C1)
        elif "START_FEEDBACK" in self.current_cmd:
            if config.CONDITION == "C0":
                self.loop_done_pub.publish(String(data=f"DONE:{self.current_cmd}"))

            elif config.CONDITION == "C1":
                if self.dialogue_count < config.MAX_EXCHANGES:
                    self.set_stt(True)
                else:
                    self.stop_silence_timer()
                    self.loop_done_pub.publish(String(data=f"DONE:{self.current_cmd}"))

            elif config.CONDITION == "C2":
                # Phrase d'invitation à s'inspirer, puis on passe au tour suivant
                self.stop_silence_timer()
                phrase = random.choice(config.C2_PHRASES_INVITE)
                self.send_robot("None", "None", phrase)
                self._after_speech(
                    lambda: self.loop_done_pub.publish(String(data=f"DONE:{self.current_cmd}"))
                )


        elif "START_TITLE" in self.current_cmd:
            self.loop_done_pub.publish(String(data=f"DONE:{self.current_cmd}"))



    def say_goodbye_final(self):
        """ Callback pour déclencher la parole après le mouvement de tête """
        if self.feedback_timer:
            self.feedback_timer.cancel()
            self.feedback_timer = None
            
        self.send_robot("bye", "happy", "Voilà, le temps est écoulé ! #LAUGH02# Merci d'avoir dessiné avec moi aujourd'hui !")

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
        msg.data = [float(yaw), float(pitch)]
        self.head_pub.publish(msg)

    def stop_silence_timer(self):
        """ Arrête proprement le timer de relance """
        if self.silence_timer:
            self.silence_timer.cancel()
            self.silence_timer = None
        
    def reset_silence_timer(self):
        """ Lance le timer de 10s seulement si on est en discussion active """
        self.stop_silence_timer()
        # On n'active la relance que si on attend vraiment une réponse
        is_dialogue = "START_FEEDBACK" in self.current_cmd or "START_ICE_BREAKING" in self.current_cmd
        
        if is_dialogue and not self.is_busy:
            # En C0, on ne veut pas de relance de silence
            if "START_FEEDBACK" in self.current_cmd and config.CONDITION == "C0":
                return
                
            self.silence_timer = self.create_timer(15.0, self.prompt_user_silence)
    
    def schedule_random_animation(self):
        """ Planifie la prochaine animation avec un delai aleatoire """
        self.stop_random_animation()
        wait_time = random.uniform(40.0, 80.0)
        self.animation_timer = self.create_timer(wait_time, self.do_random_animation)
    
    def do_random_animation(self):
        self.stop_random_animation()

        # On n'anime que si le robot ne fait rien d'autre
        if not self.is_busy and "START_DRAWING" in self.current_cmd:
            # Liste des gestes disponibles
            gestures = ["bored", "neutral", "kiss"]
          
            self.send_robot(random.choice(gestures), "None", "")
            
            def reset_head_cb():
                # On détruit le timer immédiatement pour qu'il ne tourne qu'une fois
                if hasattr(self, 'reset_timer') and self.reset_timer:
                    self.reset_timer.destroy()
                    self.reset_timer = None
                
                # Action de retour au dessin
                if "START_DRAWING" in self.current_cmd and not self.is_busy:
                    self.move_head(0, config.HEAD_PITCH_DRAWING)

            # On stocke le timer pour pouvoir le détruire proprement
            self.reset_timer = self.create_timer(4.0, reset_head_cb)

        # Replanification du prochain cycle aléatoire (40-80s)
        if "START_DRAWING" in self.current_cmd:
            self.schedule_random_animation()

    def stop_random_animation(self):
        """ Arrete proprement le timer d'animation """
        if self.animation_timer:
            self.animation_timer.cancel()
            self.animation_timer = None

    def prompt_user_silence(self):
        """ QT fait une petite relance car l'utilisateur ne dit rien """
        self.stop_silence_timer()
        
        # On ne relance que si le robot n'est pas déjà en train de parler
        if not self.is_busy:
            self.set_stt(False)
            text = config.FOLLOW_UP_QUESTION
            self.send_robot("None", "None", text)
            
            # On bloque le micro le temps de la relance
            self.is_busy = True
            self._after_speech(self.finish_feedback_loop)

def main(args=None):
    rclpy.init(args=args)
    node = InteractionNode() 

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        # Nettoyage rapide du noeud
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()
        
        # Sortie immediate pour eviter la pollution des logs ROS 2
        os._exit(0)
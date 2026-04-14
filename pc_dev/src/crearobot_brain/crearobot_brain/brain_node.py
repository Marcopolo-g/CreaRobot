import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import openai
from . import config

class BrainNode(Node):
    def __init__(self):
        super().__init__('brain_node')
        openai.api_key = config.OPENAI_API_KEY
        
        # Memoire qui s'accumule au fil des tours
        self.visual_memory = "" 
        self.chat_history = []
        self.current_phase = ""
        
        # Souscription a l'image (declenchee par InteractionNode)
        self.img_sub = self.create_subscription(String, '/pc/vision/image_raw_b64', self.process_vision, 10)
        # Souscription au texte pour la discussion continue
        self.stt_sub = self.create_subscription(String, '/pc/stt/transcript', self.handle_dialogue, 10)
        
        # Envoi de la reponse vers InteractionNode
        self.tts_pub = self.create_publisher(String, '/pc/vision/feedback', 10)

        self.phase_sub = self.create_subscription(String, '/pc/phase_control', self.phase_callback, 10)
        
        self.warmup_llm()

        self.get_logger().info("Brain Node prêt")

    def phase_callback(self, msg):
        self.current_phase = msg.data

    def warmup_llm(self):
        # On lance une requete minuscule pour ouvrir la connexion
        try:
            openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1
            )
        except Exception as e:
            self.get_logger().error(f"Echec du warmup : {e}")

    def process_vision(self, msg):
        # On commence par analyser ce qu'il y a sur l'image
        self.get_logger().info("Analyse du dessin en cours...")

        try:
            is_title_phase = "START_TITLE" in self.current_phase

            if is_title_phase:
                prompt = config.PROMPT_TITLE
            elif not self.visual_memory:
                prompt = "Décris ce que tu vois sur le dessin de maniere a pouvoir te rappeler du dessin plus tard sans le voir (entre autre, précise les formes et les objets de ce dessin)."
            else:
                prompt = f"Precedemment, le dessin etait : {self.visual_memory}."
                prompt += f"Dis moi ce qui a change ou ce qui est nouveau ou les modifications apportées par rapport à la description précédente. Sois très précis sur les ajouts."

            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{msg.data}"}}
                    ]}
                ],
                max_tokens=300
            )
            
            # Mise a jour de la memoire visuelle
            answer = response.choices[0].message.content

            if is_title_phase:
                # On envoie le titre généré et on s'arrête là !
                self.send_to_robot(answer)
                return

            if not self.visual_memory:
                # Initialisation
                self.visual_memory = answer
            else:
                # On empile les ajouts pour mettre en évidence la progression
                self.visual_memory += f"\n[Ajouts récents] : {answer}"
            
            # On aiguille selon la condition definie dans la config
            if config.CONDITION == "C1":
                self.generate_c1_feedback()
            elif config.CONDITION == "C2":
                self.generate_c2_feedback(msg.data) # On passe l'image pour C2

        except Exception as e:
            self.get_logger().error(f"Erreur Vision : {e}")

    def generate_c1_feedback(self):
        # Logique C1 : QT discute et suggere des idees
        try:
            prompt = f"{config.PROMPT_C1}\nDessin actuel : {self.visual_memory}\n"
            
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": prompt}],
                max_tokens=100
            )
            
            text = response.choices[0].message.content
            self.send_to_robot(text)
            
        except Exception as e:
            self.get_logger().error(f"Erreur C1 : {e}")

    def generate_c2_feedback(self, base64_image):
        # Logique C2 : QT va generer une image (DALL-E) plus tard ici
        # Pour l'instant on prepare juste la structure
        self.get_logger().info("Preparation du feedback C2 (Surprise)")
        
        msg_c2 = "C'est super ! J'ai une idée de génie pour transformer ton dessin, regarde !"
        # Ici on viendra inserer l'appel a l'API de generation d'image
        self.send_to_robot(msg_c2)


    def handle_dialogue(self, msg):
        # On vérifie si on est dans une phase autorisée (Feedback OU Ice Breaking)
        is_feedback = "START_FEEDBACK" in self.current_phase
        is_ice_breaking = "START_ICE_BREAKING" in self.current_phase

        if not is_feedback and not is_ice_breaking:
            return
        
        # En feedback on a besoin de la vision, en ice breaking non.
        if is_feedback and not self.visual_memory:
            return

        user_input = msg.data
        
        # On définit le prompt et le contexte selon la phase
        if is_ice_breaking:
            # Prompt simple pour briser la glace
            system_prompt = config.PROMPT_ICE_BREAKING
            messages = [{"role": "system", "content": system_prompt}]
        else:
            # Prompt classique pour le feedback du dessin
            system_prompt = config.PROMPT_C1
            messages = [{"role": "system", "content": system_prompt}]
            messages.append({"role": "system", "content": f"Memoire visuelle : {self.visual_memory}"})
        
        # On ajoute l'historique récent (valable pour les deux phases)
        messages.extend(self.chat_history[-6:])
        messages.append({"role": "user", "content": user_input})

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=80
            )
            
            answer = response.choices[0].message.content
            
            # On enregistre dans l'historique pour que QT réponde de manière cohérente
            self.chat_history.append({"role": "user", "content": user_input})
            self.chat_history.append({"role": "assistant", "content": answer})
            
            self.send_to_robot(answer)
            
        except Exception as e:
            self.get_logger().error(f"Erreur Dialogue ({self.current_phase}) : {e}")

    def send_to_robot(self, text):
        msg = String()
        msg.data = text
        self.tts_pub.publish(msg)

def main():
    rclpy.init()
    node = BrainNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
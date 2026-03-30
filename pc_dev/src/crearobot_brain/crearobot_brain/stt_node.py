# stt_node.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from faster_whisper import WhisperModel
import numpy as np
from std_msgs.msg import UInt8MultiArray 


class STTNode(Node):
    def __init__(self):
        super().__init__('stt_node')
        
        # Modèle Whisper local (tiny = rapide, base = plus précis ou small)
        self.get_logger().info("Chargement du modèle Whisper...")
        self.model = WhisperModel("small", device="cpu", compute_type="int8")
        self.get_logger().info("Modèle prêt !")

        # Buffer audio
        self.audio_buffer = []
        self.is_recording = False
        self.silence_counter = 0
        self.SILENCE_LIMIT = 20   # ~20 chunks = ~0.4s de silence avant de couper
        self.MIN_AUDIO_LENGTH = 10  # Minimum de chunks pour éviter les faux positifs
        self.ENERGY_THRESHOLD = 400  # Seuil de détection de voix

        # Subscriber : reçoit l'audio brut depuis la gateway (qui le bridge depuis le robot)
        self.sub_audio = self.create_subscription(UInt8MultiArray, '/pc/raw_audio', self.audio_callback, 100)

        # Publisher : envoie le texte reconnu vers brain_node
        self.pub_text = self.create_publisher(String, '/pc/user_speech', 10)

        self.get_logger().info("STT Node prêt, en écoute sur /pc/raw_audio")


    def audio_callback(self, msg):
        # 1. Conversion des octets reçus en entiers
        audio_bytes = bytes(msg.data)
        raw_data = np.frombuffer(audio_bytes, dtype=np.int16)

        # 2. Séparation des 6 canaux (Reshape)
        # On transforme le vecteur plat en une matrice [Nombre_de_frames, 6]
        try:
            audio_matrix = raw_data.reshape(-1, 6)
            # On ne garde que le canal 0 (le micro de devant)
            audio_chunk = audio_matrix[:, 0]
        except Exception as e:
            # Si le buffer est mal coupé, on ignore ce chunk pour éviter le crash
            return

        # 3. Calcul de l'énergie sur le canal 0 uniquement
        energy = np.abs(audio_chunk).mean()

        if energy > self.ENERGY_THRESHOLD:
            if not self.is_recording:
                self.get_logger().info("Voix détectée sur Canal 0, enregistrement...")
                self.is_recording = True
            # On ajoute le canal 0 au buffer (pas les 6 !)
            self.audio_buffer.extend(audio_chunk.tolist())
            self.silence_counter = 0

        elif self.is_recording:
            self.audio_buffer.extend(audio_chunk.tolist())
            self.silence_counter += 1
            
            if self.silence_counter >= self.SILENCE_LIMIT:
                if len(self.audio_buffer) > self.MIN_AUDIO_LENGTH * 1024:
                    self.transcribe()
                self.audio_buffer = []
                self.is_recording = False
                self.silence_counter = 0

    def transcribe(self):
        self.get_logger().info(" Transcription en cours (Whisper)...") # Ajoute ça
        audio_np = np.array(self.audio_buffer, dtype=np.float32) / 32768.0
        
        segments, _ = self.model.transcribe(
            audio_np,
            language="fr",
            beam_size=1
        )
        
        text = " ".join([seg.text for seg in segments]).strip()
        
        if text:
            self.get_logger().info(f" RECONNU : '{text}'") # Ce log s'affichera dans ton terminal de run
            msg = String()
            msg.data = text
            self.pub_text.publish(msg)
        else:
            self.get_logger().info(" Transcription terminée, mais aucun texte détecté.")

def main(args=None):
    rclpy.init(args=args)
    node = STTNode()
    rclpy.spin(node)
    rclpy.shutdown()
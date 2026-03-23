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
        
        # Modèle Whisper local (tiny = rapide, base = plus précis)
        self.get_logger().info("Chargement du modèle Whisper...")
        self.model = WhisperModel("tiny", device="cpu", compute_type="int8")
        self.get_logger().info("Modèle prêt !")

        # Buffer audio
        self.audio_buffer = []
        self.is_recording = False
        self.silence_counter = 0
        self.SILENCE_LIMIT = 20   # ~20 chunks = ~0.4s de silence avant de couper
        self.MIN_AUDIO_LENGTH = 10  # Minimum de chunks pour éviter les faux positifs
        self.ENERGY_THRESHOLD = 300  # Seuil de détection de voix

        # Subscriber : reçoit l'audio brut depuis la gateway (qui le bridge depuis le robot)
        self.sub_audio = self.create_subscription(UInt8MultiArray, '/pc/raw_audio', self.audio_callback, 100)

        # Publisher : envoie le texte reconnu vers brain_node
        self.pub_text = self.create_publisher(String, '/pc/user_speech', 10)

        self.get_logger().info("STT Node prêt, en écoute sur /pc/raw_audio")

    def audio_callback(self, msg):
        # Convertir les bytes en numpy
        audio_chunk = np.array(msg.data, dtype=np.int16)
        energy = np.abs(audio_chunk).mean()

        if energy > self.ENERGY_THRESHOLD:
            # Voix détectée
            if not self.is_recording:
                self.get_logger().info("Voix détectée, enregistrement...")
                self.is_recording = True
            self.audio_buffer.extend(audio_chunk.tolist())
            self.silence_counter = 0
        elif self.is_recording:
            # Silence après une voix
            self.audio_buffer.extend(audio_chunk.tolist())
            self.silence_counter += 1

            if self.silence_counter >= self.SILENCE_LIMIT:
                # Fin de phrase détectée
                if len(self.audio_buffer) > self.MIN_AUDIO_LENGTH * 1024:
                    self.transcribe()
                self.audio_buffer = []
                self.is_recording = False
                self.silence_counter = 0

    def transcribe(self):
        audio_np = np.array(self.audio_buffer, dtype=np.float32) / 32768.0
        
        segments, _ = self.model.transcribe(
            audio_np,
            language="fr",
            beam_size=1,      # Rapide
            best_of=1,
            temperature=0.0
        )
        
        text = " ".join([seg.text for seg in segments]).strip()
        
        if text:
            self.get_logger().info(f"Reconnu : {text}")
            msg = String()
            msg.data = text
            self.pub_text.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = STTNode()
    rclpy.spin(node)
    rclpy.shutdown()
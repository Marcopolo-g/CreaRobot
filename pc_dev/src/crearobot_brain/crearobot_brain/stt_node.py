import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from faster_whisper import WhisperModel
import numpy as np
import sounddevice as sd


class STTNode(Node):
    def __init__(self):
        super().__init__('stt_node')
        self.pub = self.create_publisher(String, '/pc/stt/transcript', 10)
        self.create_subscription(Bool, 'pc/stt/enable', self.enable_callback, 10)

        self.enabled = False
        self.audio_buffer = [] # On utilise une liste simple pour accumuler
        
        self.get_logger().info("Chargement de Whisper BASE sur CPU...")
        
        try:
            # On force le CPU et le mode INT8
            self.model = WhisperModel("base", device="cpu", compute_type="int8")
            self.get_logger().info("--- WHISPER CPU PRET ---")
        except Exception as e:
            self.get_logger().error(f"Erreur : {e}")

    def enable_callback(self, msg):
        if self.enabled != msg.data:
            self.enabled = msg.data
            if self.enabled:
                self.audio_buffer = [] # Reset du son quand on active
                self.get_logger().info("Micro ACTIF - Parlez maintenant...")
            else:
                self.get_logger().info("Micro INACTIF")

    def audio_callback(self, indata, frames, time, status):
        if self.enabled:
            self.audio_buffer.append(indata.copy())

    def process_live(self):
        """Analyse ce qui a ete capture jusqu'a present"""
        if not self.audio_buffer:
            return

        # On transforme la liste en tableau numpy
        recording = np.concatenate(self.audio_buffer, axis=0).flatten()
        
        # On ne lance l'analyse que si on a au moins 0.5 seconde de son
        if len(recording) < 16000 * 0.5:
            return

        try:
            # vad_filter=True est essentiel ici pour separer les phrases
            segments, info = self.model.transcribe(recording, language="fr", vad_filter=True)
            
            full_text = ""
            last_end = 0
            for segment in segments:
                full_text += segment.text
                last_end = segment.end # Fin du dernier mot detecte

            # CALCUL DE LA PAUSE :
            # Si le dernier mot fini depuis plus de 0.8 seconde, on considere la phrase finie
            duration_total = len(recording) / 16000
            silence_at_end = duration_total - last_end

            if full_text.strip() and silence_at_end > 0.8:
                final_text = full_text.strip()
                self.get_logger().info(f"Phrase detectee : {final_text}")
                
                msg = String()
                msg.data = final_text
                self.pub.publish(msg)
                
                # On vide le buffer pour la phrase suivante
                self.audio_buffer = []
                
        except Exception as e:
            self.get_logger().error(f"Erreur transcription : {e}")

def main(args=None):
    rclpy.init(args=args)
    node = STTNode()

    # Flux audio ouvert en continu
    with sd.InputStream(samplerate=16000, channels=1, callback=node.audio_callback):
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            
            # On analyse en continu pendant que le micro est ouvert
            if node.enabled:
                node.process_live()

    node.destroy_node()
    rclpy.shutdown()
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, UInt8MultiArray
import numpy as np
import speech_recognition as sr
import io
import wave

class STTNode(Node):
    def __init__(self):
        super().__init__('stt_node')
        
        # Initialisation du reconnaisseur
        self.recognizer = sr.Recognizer()
        self.get_logger().info("Google STT prêt !")

        self.audio_buffer = []
        self.is_recording = False
        self.silence_counter = 0
        self.SILENCE_LIMIT = 15    # ~0.3s
        self.ENERGY_THRESHOLD = 200 # À ajuster selon les logs d'énergie

        self.sub_audio = self.create_subscription(UInt8MultiArray, '/pc/raw_audio', self.audio_callback, 100)
        self.pub_text = self.create_publisher(String, '/pc/user_speech', 10)

    def audio_callback(self, msg):
        audio_chunk = np.frombuffer(bytes(msg.data), dtype=np.int16)
        energy = np.abs(audio_chunk).mean()
        

        if energy > self.ENERGY_THRESHOLD:
            if not self.is_recording:
                self.get_logger().info("Écoute (Google)...")
                self.is_recording = True
            self.audio_buffer.extend(bytes(msg.data))
            self.silence_counter = 0
        elif self.is_recording:
            self.audio_buffer.extend(bytes(msg.data))
            self.silence_counter += 1
            if self.silence_counter >= self.SILENCE_LIMIT:
                self.transcribe_free_google()
                self.audio_buffer = []
                self.is_recording = False
                self.silence_counter = 0

    def transcribe_free_google(self):
        self.get_logger().info("Envoi à Google STT...")
        
        # Conversion du buffer en objet AudioData pour speech_recognition
        raw_data = bytes(self.audio_buffer)

        with open("debug_audio.wav", "wb") as f:
            with wave.open(f, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(raw_data)
        self.get_logger().info("Fichier debug_audio.wav enregistré. Écoute-le !")
        
        # On crée un faux fichier WAV en mémoire pour que Google comprenne le format
        with io.BytesIO() as wav_file:
            with wave.open(wav_file, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2) # 16-bit
                wf.setframerate(16000)
                wf.writeframes(raw_data)
            wav_file.seek(0)
            
            with sr.AudioFile(wav_file) as source:
                audio_data = self.recognizer.record(source)

        try:
            # L'API magique sans clé
            text = self.recognizer.recognize_google(audio_data, language="fr-FR")
            self.get_logger().info(f"RECONNU : '{text}'")
            
            msg = String()
            msg.data = text
            self.pub_text.publish(msg)
            
        except sr.UnknownValueError:
            self.get_logger().info("Google n'a pas compris l'audio.")
        except sr.RequestError as e:
            self.get_logger().error(f"Erreur réseau Google STT : {e}")

def main(args=None):
    rclpy.init(args=args)
    node = STTNode()
    rclpy.spin(node)
    rclpy.shutdown()
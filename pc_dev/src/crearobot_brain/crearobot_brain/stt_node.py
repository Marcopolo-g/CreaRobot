import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import speech_recognition as sr
import threading
import os
from std_msgs.msg import Bool, String
import time

os.environ['AS_SILENCE_ALSA'] = '1'

class STTNode(Node):
    def __init__(self):
        super().__init__('stt_node')
        self.pub = self.create_publisher(String, '/pc/stt/transcript', 10)

        # Le STT est désactivé par défaut au démarrage
        self.enabled = False
        self.create_subscription(Bool, 'pc/stt/enable', self.enable_callback, 10)

        self.recognizer = sr.Recognizer()

        # Réglages pour éviter de capter le "silence bruyant"
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.energy_threshold = 400 
        self.recognizer.pause_threshold = 0.8
        self.recognizer.phrase_threshold = 0.3 # Évite de déclencher sur des bruits de 0.1s
        self.recognizer.non_speaking_duration = 0.2

        self.mic = sr.Microphone()
        self.thread = threading.Thread(target=self.listen_loop, daemon=True)
        self.thread.start()

        self.get_logger().info("STT Node prêt et sécurisé")

    def enable_callback(self, msg):
        self.enabled = msg.data
        status = "ACTIF" if self.enabled else "INACTIF"
        self.get_logger().info(f"État du STT : {status}")

    def listen_loop(self):
        with self.mic as source:
            self.get_logger().info("Calibration...")
            # On augmente un peu la calibration pour mieux filtrer le bruit des moteurs/ventilos
            self.recognizer.adjust_for_ambient_noise(source, duration=1.5)
            
            while rclpy.ok():
                if not self.enabled:
                    time.sleep(0.2)
                    continue
                try:
                    self.get_logger().info("Écoute...")
                    audio = self.recognizer.listen(source, timeout=None, phrase_time_limit=10)

                    self.get_logger().info("Analyse...")
                    text = self.recognizer.recognize_google(audio, language="fr-FR")

                    # NETTOYAGE ET VALIDATION
                    if text:
                        clean_text = text.strip()
                        if len(clean_text) > 0:
                            self.get_logger().info(f"RÉSULTAT : {clean_text}")
                            self.pub.publish(String(data=clean_text))
                        else:
                            self.get_logger().warn("Texte vide après nettoyage, pas d'envoi.")
                    
                except sr.UnknownValueError:
                    # On réduit le log pour ne pas polluer, mais on ne publie rien
                    self.get_logger().info("Bruit détecté mais aucun mot reconnu.")
                    continue
                except sr.RequestError as e:
                    self.get_logger().error(f"Erreur réseau Google : {e}")
                except Exception as e:
                    self.get_logger().error(f"Erreur imprévue : {e}")

def main(args=None):
    rclpy.init(args=args)
    node = STTNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
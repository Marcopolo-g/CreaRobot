import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from faster_whisper import WhisperModel
import numpy as np
import sounddevice as sd
import time


class STTNode(Node):
    def __init__(self):
        super().__init__('stt_node')
        self.pub = self.create_publisher(String, '/pc/stt/transcript', 10)
        self.create_subscription(Bool, '/pc/stt/enable',      self.enable_callback,   10)
        self.create_subscription(Bool, '/pc/robot/speaking',  self.speaking_callback, 10)

        self.enabled         = False
        self.robot_speaking  = False  # prioritaire sur enabled
        self.audio_buffer    = []
        self._mic_open_logged = False

        try:
            self.model = WhisperModel("base", device="cuda", compute_type="float16")
            self.get_logger().info("--- WHISPER GPU PRET ---")
        except Exception:
            self.get_logger().warning("GPU non disponible — fallback CPU")
            self.model = WhisperModel("base", device="cpu", compute_type="int8")
            self.get_logger().info("--- WHISPER CPU PRET ---")

    @property
    def _listening(self):
        return self.enabled and not self.robot_speaking

    def enable_callback(self, msg):
        self.enabled = msg.data
        if not msg.data:
            self.audio_buffer = []

    def speaking_callback(self, msg):
        self.robot_speaking = msg.data
        if msg.data:
            self.audio_buffer = []
            self._mic_open_logged = False
        elif self.enabled and not self._mic_open_logged:
            self._mic_open_logged = True
            self.get_logger().info("Micro ouvert — en écoute")

    def audio_callback(self, indata, frames, time, status):
        if self._listening:
            self.audio_buffer.append(indata.copy())

    def process_live(self):
        if not self._listening or not self.audio_buffer:
            return

        recording = np.concatenate(self.audio_buffer, axis=0).flatten()
        if len(recording) < 16000 * 0.5:
            return

        try:
            t0 = time.time()
            segments, _ = self.model.transcribe(recording, language="fr", vad_filter=True)

            full_text = ""
            last_end  = 0
            for segment in segments:
                full_text += segment.text
                last_end   = segment.end

            duration_total  = len(recording) / 16000
            silence_at_end  = duration_total - last_end

            if full_text.strip() and silence_at_end > 0.8:
                self.get_logger().info(f"[LATENCE] Whisper transcription : {time.time() - t0:.2f} s")
                final_text = full_text.strip()
                self.get_logger().info(f"Phrase detectee : {final_text}")
                msg = String()
                msg.data = final_text
                self.pub.publish(msg)
                self.audio_buffer = []

        except Exception as e:
            self.get_logger().error(f"Erreur transcription : {e}")


def main(args=None):
    rclpy.init(args=args)
    node = STTNode()

    with sd.InputStream(samplerate=16000, channels=1, callback=node.audio_callback):
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            node.process_live()

    node.destroy_node()
    rclpy.shutdown()

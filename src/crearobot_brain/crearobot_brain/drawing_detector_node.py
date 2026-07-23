import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
import cv2
import numpy as np
import json
from collections import deque

from . import config


class DrawingDetectorNode(Node):

    def __init__(self):
        super().__init__('drawing_detector_node')

        self._active        = False
        self._cap           = None
        self._prev_frame    = None
        self._still_secs    = 0.0
        self._timer         = None
        self._warmup_timer  = None
        self._diff_history  = deque(maxlen=config.FRAME_DIFF_WINDOW_SIZE)

        self.pub = self.create_publisher(Bool, '/pc/vision/drawing_stopped', 10)
        self.create_subscription(String, '/pc/phase_control', self._phase_cb, 10)

        self.get_logger().info("DrawingDetector prêt — en attente d'une phase START_DRAWING")

    # ── Phase callback ──────────────────────────────────────────────────────────

    def _phase_cb(self, msg):
        try:
            data = json.loads(msg.data)
        except Exception:
            return
        phase = data.get("phase", "")
        is_drawing = "START_DRAWING" in phase

        if is_drawing and not self._active:
            self._start()
        elif not is_drawing and self._active:
            self._stop()

    # ── Start / Stop ────────────────────────────────────────────────────────────

    def _start(self):
        self._cap = cv2.VideoCapture(config.CAMERA_INDEX)
        if not self._cap.isOpened():
            self.get_logger().error("DrawingDetector : caméra inaccessible")
            self._cap = None
            return
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._active     = True
        self._prev_frame = None
        self._still_secs = 0.0
        self._diff_history.clear()
        self.get_logger().info("DrawingDetector : caméra ouverte — détection dans 20s")
        self._warmup_timer = self.create_timer(config.FRAME_DIFF_WARMUP, self._start_detection)

    def _start_detection(self):
        self._warmup_timer.cancel()
        self._warmup_timer = None
        self._timer = self.create_timer(config.FRAME_DIFF_INTERVAL, self._tick)
        self.get_logger().info("DrawingDetector activé")

    def _stop(self):
        if self._warmup_timer is not None:
            self._warmup_timer.cancel()
            self._warmup_timer = None
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._active     = False
        self._prev_frame = None
        self._still_secs = 0.0
        self.get_logger().info("DrawingDetector désactivé")

    # ── Detection tick ──────────────────────────────────────────────────────────

    def _tick(self):
        if self._cap is None or not self._cap.isOpened():
            return

        # Vide le buffer pour obtenir la frame la plus récente
        for _ in range(5):
            self._cap.grab()
        ret, frame = self._cap.retrieve()
        if not ret:
            self.get_logger().warning("DrawingDetector : frame manquée")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self._prev_frame is None:
            self._prev_frame = gray
            return

        diff      = cv2.absdiff(gray, self._prev_frame)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        moved_px  = int(np.count_nonzero(thresh))
        self._prev_frame = gray

        self._diff_history.append(moved_px)
        median_px = float(np.median(self._diff_history))

        if median_px >= config.FRAME_DIFF_PIXEL_THRESHOLD:
            self._still_secs = 0.0
            #self.get_logger().info(
            #    f"[DIFF] mouvement : {moved_px:>6} px  médiane={median_px:.0f} → reset"
            #)
        else:
            self._still_secs += config.FRAME_DIFF_INTERVAL
            #self.get_logger().info(
            #    f"[DIFF] immobile  : {moved_px:>6} px  médiane={median_px:.0f} — {self._still_secs:.1f}s/{config.FRAME_DIFF_INACTIVITY_DURATION:.1f}s"
            #)
            if self._still_secs >= config.FRAME_DIFF_INACTIVITY_DURATION:
                self.get_logger().info(
                    f"DrawingDetector : fin de dessin détectée ({self._still_secs:.1f}s sans mouvement)"
                )
                msg = Bool()
                msg.data = True
                self.pub.publish(msg)
                self._still_secs = 0.0


def main(args=None):
    rclpy.init(args=args)
    node = DrawingDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._stop()
        node.destroy_node()
        rclpy.shutdown()

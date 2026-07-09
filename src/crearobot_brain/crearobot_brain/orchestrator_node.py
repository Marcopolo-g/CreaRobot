import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32, Bool
import json
import sys
import threading
import os

from . import config

# orchestrator_node.py
class OrchestratorNode(Node):
    def __init__(self):
        super().__init__('orchestrator_node')
        
        self.current_loop = 1
        self.state = "INTRO" # INTRO, DRAWING, FEEDBACK, ENDING
        
        # Publie la phase et le tour actuel
        self.phase_ctrl_pub = self.create_publisher(String, '/pc/phase_control', 10)
        self.tour_ctrl_pub = self.create_publisher(Int32, '/pc/tour_control', 10)

        self.interaction_finished_sub = self.create_subscription(String, '/pc/interaction_finished', self.on_interaction_done, 10)
        self.speech_sub = self.create_subscription(String, '/pc/stt/transcript', self.state_machine, 10)
        self.create_subscription(Bool, '/pc/vision/drawing_stopped',    self.on_drawing_stopped,    10)
        self.create_subscription(Bool, '/pc/brain/addressing_detected', self.on_addressing_detected, 10)

        self.phase_timer = None
        self.init_timer = self.create_timer(2.0, self.start_experience)

        self.input_thread = threading.Thread(target=self.terminal_listener, daemon=True)
        self.input_thread.start()

    def terminal_listener(self):
        while rclpy.ok():
            user_input = sys.stdin.readline().strip().lower()
            if not user_input:
                break
            if user_input == "fini":
                self.stop_timer()
                self.get_logger().warn("FINI terminal : passage en FEEDBACK/TITLE")
                if self.current_loop >= config.MAX_LOOPS:
                    self.change_state("TITLE")
                else:
                    self.change_state("FEEDBACK")
            elif user_input == "terminate":
                self.stop_timer()
                self.get_logger().warn("TERMINATE terminal : passage en TITLE")
                self.change_state("TITLE")
            if user_input == "skip":
                self.stop_timer()
                if self.current_loop >= config.MAX_LOOPS:
                    self.get_logger().warn(f"SKIP : passage direct en TITLE depuis {self.state}")
                    self.change_state("TITLE")
                else:
                    self.get_logger().warn(f"SKIP : passage direct en FEEDBACK depuis {self.state}")
                    self.change_state("FEEDBACK")
            elif user_input == "title":
                self.stop_timer()
                self.get_logger().warn("TITLE forcé depuis le terminal")
                self.change_state("TITLE")

    def start_experience(self):
        if self.init_timer:
            self.init_timer.cancel()
            self.init_timer = None

        if config.DEBUG_SKIP_TO_FEEDBACK:
            self.get_logger().warn("DEBUG_SKIP_TO_FEEDBACK activé : démarrage direct en DRAWING")
            self.change_state("DRAWING")
        else:
            self.change_state("INTRO")

    def state_machine(self, msg):
        text = msg.data.lower()

        if self.state == "INTRO" and any(x in text for x in ["oui", "pret", "bon"]):
            self.change_state("ICE_BREAKING")

    def change_state(self, new_state, triggered_by=""):
        self.state = new_state

        payload = {
            "phase": f"START_{new_state}",
            "tour": self.current_loop,
        }
        if triggered_by:
            payload["triggered_by"] = triggered_by

        if self.state == "DRAWING":
            self.start_draw_timer()

        msg = String()
        msg.data = json.dumps(payload)
        self.phase_ctrl_pub.publish(msg)

        tour_msg = Int32()
        tour_msg.data = self.current_loop
        self.tour_ctrl_pub.publish(tour_msg)
        self.get_logger().info(f"ETAT : {self.state} (Tour {self.current_loop})")

    def start_draw_timer(self):
        self.stop_timer()
        self.phase_timer = self.create_timer(config.DRAW_DURATION, self.on_draw_timeout)

    def on_drawing_stopped(self, msg):
        if msg.data and self.state == "DRAWING":
            self.get_logger().info("DrawingDetector : fin de dessin détectée — transition de phase")
            self.on_draw_timeout()

    def on_addressing_detected(self, msg):
        if msg.data and self.state == "DRAWING":
            self.get_logger().info("Adressage détecté — transition vers feedback")
            self.stop_timer()
            if self.current_loop >= config.MAX_LOOPS:
                self.change_state("TITLE")
            else:
                self.change_state("FEEDBACK", triggered_by="addressing")

    def on_draw_timeout(self):
        self.stop_timer()

        # SI on est au dernier tour, on ne fait pas de feedback, on finit !
        if self.current_loop >= config.MAX_LOOPS:
            self.change_state("TITLE")
        else:
            # Sinon on passe au feedback normalement
            self.change_state("FEEDBACK")

    def on_interaction_done(self, msg):
        """ QT a fini son feedback : on lance toujours le tour suivant """
        parts = msg.data.split(":")
        status = parts[0]  # "DONE"
        phase  = parts[1] if len(parts) > 1 else ""

        if status != "DONE":
            return

        if self.state == "ICE_BREAKING" and "ICE_BREAKING" in phase:
            self.change_state("TASK_INTRO")

        elif self.state == "TASK_INTRO" and "TASK_INTRO" in phase:
            self.change_state("DRAWING")

        elif self.state == "DRAWING" and "DRAWING" in phase:
            self.stop_timer()
            if self.current_loop >= config.MAX_LOOPS:
                self.change_state("TITLE")
            else:
                self.change_state("FEEDBACK")

        elif self.state == "FEEDBACK" and "FEEDBACK" in phase:
            self.current_loop += 1
            self.change_state("DRAWING")

        elif self.state == "TITLE" and "TITLE" in phase:
            self.change_state("ENDING")


    def stop_timer(self):
        if self.phase_timer:
            self.phase_timer.cancel()
            self.phase_timer = None


def main(args=None):
    rclpy.init(args=args)
    node = OrchestratorNode()
    
    # On utilise un MultiThreadedExecutor pour que les timers et le thread clavier
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        print("\n[Orchestrator] Arrêt demandé par l'utilisateur...")
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()
        
        # On force la fermeture de tous les threads (dont le thread clavier)
        os._exit(0)
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32
import threading

from . import config

class OrchestratorNode(Node):
    def __init__(self):
        super().__init__('orchestrator_node')
        
        self.current_phase = 1
        self.phase_timer = None
        self.start_timer = None
        
        # TIMERS
        self.DURATIONS = {2: config.TIME_PHASE_2, 4: config.TIME_PHASE_4}
        
        # Publishers : On envoie juste l'ordre de la phase
        self.phase_ctrl_pub = self.create_publisher(String, '/pc/phase_control', 10)
        
        # Subscribers : On appelle speech_callback pour "nettoyer" le message ROS
        self.speech_sub = self.create_subscription(String, '/pc/user_speech', self.speech_callback, 10)

        self.get_logger().info("=== Orchestrateur prêt ===")
        
        # Timer de démarrage (One-shot)
        self.start_timer = self.create_timer(2.0, self.start_experience)

        # Thread terminal
        self.thread = threading.Thread(target=self.terminal_input_loop)
        self.thread.daemon = True 
        self.thread.start()

    def terminal_input_loop(self):
        """ Boucle pour le terminal """
        while rclpy.ok():
            try:
                raw_text = input(f"\n[PHASE {self.current_phase}] > ")
                self.state_machine(raw_text)
            except EOFError:
                break

    def speech_callback(self, msg):
        """ Callback pour le micro : on extrait le texte et on l'envoie à la machine à états """
        self.state_machine(msg.data)

    def state_machine(self, text):
        """ Logique de transition universelle (reçoit toujours du texte brut) """
        text = text.lower()
        
        if self.current_phase == 1 and any(x in text for x in ["pret", "oui"]):
            self.change_phase(2)
        elif self.current_phase == 2 and any(x in text for x in ["fini", "termine"]):
            self.stop_timer()
            self.change_phase(3)
        elif self.current_phase == 3 and any(x in text for x in ["fini", "termine", "merci", "ok"]):
            self.change_phase(4)
        elif self.current_phase == 4 and any(x in text for x in ["fini", "termine", "au revoir"]):
            self.stop_timer()
            self.change_phase(5)

    def start_experience(self):
        if self.start_timer:
            self.start_timer.cancel()
            self.start_timer = None
        self.change_phase(1)

    def change_phase(self, new_phase):
        self.current_phase = new_phase
        msg = String()
        msg.data = f"START_PHASE_{new_phase}"
        self.phase_ctrl_pub.publish(msg)
        self.get_logger().info(f"Transition vers PHASE {new_phase}")
        
        if new_phase in self.DURATIONS:
            # On s'assure d'arrêter l'ancien timer avant d'en créer un nouveau
            self.stop_timer()
            self.phase_timer = self.create_timer(self.DURATIONS[new_phase], self.timer_callback)

    def timer_callback(self):
        self.get_logger().info(f"Timeout Phase {self.current_phase}")
        self.stop_timer()
        if self.current_phase == 2: self.change_phase(3)
        elif self.current_phase == 4: self.change_phase(5)

    def stop_timer(self):
        if self.phase_timer:
            self.phase_timer.cancel()
            self.phase_timer = None

def main(args=None):
    rclpy.init(args=args)
    node = OrchestratorNode()
    # On utilise souvent MultiThreadedExecutor quand on mélange Threads et Timers en ROS2
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
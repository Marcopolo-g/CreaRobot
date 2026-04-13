import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32
import time
import threading

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

        self.phase_timer = None
        self.init_timer = self.create_timer(2.0, self.start_experience)

    def start_experience(self):
        if self.init_timer:
            self.init_timer.cancel()
            self.init_timer = None

        self.change_state("INTRO")

    def state_machine(self, msg):
        text = msg.data.lower()
        
        if self.state == "INTRO":
            time.sleep(3) # 4 secs delay before moving on to the next phase
            self.change_state("DRAWING")
            
        elif self.state == "DRAWING" and any(x in text for x in ["fini", "termine"]):
            self.stop_timer()
        
            if self.current_loop >= config.MAX_LOOPS:
                self.change_state("ENDING")
            else:
                self.change_state("FEEDBACK")

    def change_state(self, new_state):
        self.state = new_state
        msg = String()
        
        if self.state == "INTRO":
            msg.data = "START_INTRO"
        elif self.state == "DRAWING":
            msg.data = f"START_DRAWING_{self.current_loop}"
            self.start_draw_timer()
        elif self.state == "FEEDBACK":
            msg.data = f"START_FEEDBACK_{self.current_loop}"
        elif self.state == "ENDING":
            msg.data = "START_ENDING"
            
        self.phase_ctrl_pub.publish(msg)
        tour_msg = Int32()
        tour_msg.data = self.current_loop
        self.tour_ctrl_pub.publish(tour_msg)
        self.get_logger().info(f"ETAT : {self.state} (Tour {self.current_loop})")

    def start_draw_timer(self):
        self.stop_timer()
        self.phase_timer = self.create_timer(config.DRAW_DURATION, self.on_draw_timeout)

    def on_draw_timeout(self):
        self.get_logger().info(f"Fin du temps de dessin pour le tour {self.current_loop}")
        self.stop_timer()

        # SI on est au dernier tour, on ne fait pas de feedback, on finit !
        if self.current_loop >= config.MAX_LOOPS:
            self.get_logger().info("Dernier tour atteint. Passage à la conclusion.")
            self.change_state("ENDING")
        else:
            # Sinon on passe au feedback normalement
            self.change_state("FEEDBACK")

    def on_interaction_done(self, msg):
        """ QT a fini son feedback : on lance toujours le tour suivant """
        # On ne réagit que si on est en phase Feedback et qu'on reçoit "DONE"
        if self.state == "FEEDBACK" and msg.data == "DONE":
            self.get_logger().info(f"Fin du feedback pour le tour {self.current_loop}")
            
            # On passe au tour suivant et on relance le dessin
            self.current_loop += 1
            self.change_state("DRAWING")


    def stop_timer(self):
        if self.phase_timer:
            self.phase_timer.cancel()
            self.phase_timer = None


def main(args=None):
    rclpy.init(args=args)
    node = OrchestratorNode()
    # On utilise souvent MultiThreadedExecutor car on mélange Threads et Timers en ROS2
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
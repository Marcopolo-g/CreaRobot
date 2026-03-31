import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

def main():
    # --- CONFIGURATION ---
    # Ton chemin d'image spécifique
    img_path = os.path.expanduser("~/Desktop/CreaRobot/Taches_Marco.jpeg")
    topic_name = '/pc/projector/image'

    rclpy.init()
    node = Node('manual_image_pub')
    publisher = node.create_publisher(Image, topic_name, 10)
    bridge = CvBridge()

    # On attend un tout petit peu que la connexion se fasse
    import time
    time.sleep(1)

    if not os.path.exists(img_path):
        print(f"Erreur : Le fichier est introuvable ici : {img_path}")
        return

    # Lecture de l'image
    cv_image = cv2.imread(img_path)
    if cv_image is None:
        print("Erreur : Impossible de lire l'image (format corrompu ?)")
        return

    # Conversion et Publication
    msg = bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
    
    print(f"Envoi de l'image vers le projecteur...")
    publisher.publish(msg)
    
    # On laisse le temps au message de partir avant de fermer
    time.sleep(1)
    
    print("C'est envoyé ! Regarde le projecteur.")
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
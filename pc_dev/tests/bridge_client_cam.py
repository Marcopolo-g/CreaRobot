import roslibpy
import cv2
import numpy as np
import base64

# Connexion au robot
client = roslibpy.Ros(host='192.168.100.1', port=9091)

def callback(message):
    try:
        # On récupère les dimensions
        height = message['height']
        width = message['width']
        
        # On DÉCODE le Base64 (le texte devient des bytes)
        img_bytes = base64.b64decode(message['data'])
        
        # On transforme ça en tableau NumPy
        data = np.frombuffer(img_bytes, dtype=np.uint8)
        
        # On remet l'image en forme (Hauteur, Largeur, 3 couleurs)
        # Si ça crash ici, c'est que l'encodage n'est pas 'rgb8'
        image = data.reshape((height, width, 3))
        
        # ROS (RGB) -> OpenCV (BGR)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        cv2.imshow("CreaRobot", image_bgr)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            client.terminate()
            
    except Exception as e:
        print(f"Erreur de décodage : {e}")

# On s'abonne
listener = roslibpy.Topic(client, '/camera/color/image_raw', 'sensor_msgs/Image')
listener.subscribe(callback)

print("Connexion OK. Décodage du flux en cours...")
try:
    client.run_forever()
except KeyboardInterrupt:
    client.terminate()
    cv2.destroyAllWindows()
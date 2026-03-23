import roslibpy
import time

# Connexion au robot via le port 9091 (celui qu'on a ouvert)
client = roslibpy.Ros(host='192.168.100.1', port=9091)
client.run()

# Déclaration des services ROS 1 du robot pour ton PC
# On indique le nom du service et son type exact
talk_service = roslibpy.Service(client, '/qt_robot/behavior/talkText', 'qt_robot_interface/behavior_talk_text')
gesture_service = roslibpy.Service(client, '/qt_robot/gesture/play', 'qt_robot_interface/gesture_play')
emotion_service = roslibpy.Service(client, '/qt_robot/emotion/show', 'qt_robot_interface/emotion_show')

def envoyer_ordre_au_robot(geste, emotion, texte):
    print(f"Envoi au robot : [{geste}, {emotion}, {texte}]")
    
    # Préparation des données (format dictionnaire Python = JSON)
    req_talk = roslibpy.ServiceRequest({'message': texte})
    req_gesture = roslibpy.ServiceRequest({'name': 'QT/' + geste, 'speed': 0})
    req_emotion = roslibpy.ServiceRequest({'name': 'QT/' + emotion})

    # EXÉCUTION (On lance tout en même temps pour la synchro)
    # .call() envoie l'ordre au robot via le réseau
    talk_service.call(req_talk)
    gesture_service.call(req_gesture)
    emotion_service.call(req_emotion)

# --- TEST DIRECT ---
if client.is_connected:
    envoyer_ordre_au_robot("hi", "happy", "Salut ! Je te parle depuis mon PC en ROS 2 !")

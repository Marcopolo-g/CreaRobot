# Projet Hyppolite - Orchestration TCT-DP & IA Générative

> QTrobot V1 (ROS 1 Kinetic) transformé en compagnon social intelligent, piloté par des LLM et de la vision artificielle sous ROS 2.

---

## Architecture du système

L'intelligence est entièrement déportée sur un PC externe (ROS 2 / Python 3) pour pallier les limitations matérielles du robot. Le corps du robot reste piloté sous ROS 1.

```
┌─────────────────────────────────────────────────────────┐
│                     PC (ROS 2)                          │
│                                                         │
│  orchestrator_node  →  interaction_node                 │
│       ↓                      ↓                          │
│  llm_reactif_node   →   gateway_node  ← camera_node     │
└──────────────────────────┬──────────────────────────────┘
                           │ WebSocket (rosbridge :9091)
┌──────────────────────────┴──────────────────────────────┐
│                   QTRobot (ROS 1)                       │
│                                                         │
│        google SST  ·  ears.py  ·  caméra                │
└─────────────────────────────────────────────────────────┘
```

| Couche | Node | Rôle |
|---|---|---|
| Logique | `orchestrator_node` | Machine à états — 5 phases TCT-DP |
| Dispatcher | `interaction_node` | Traduit les phases en gestes / émotions / paroles |
| Cognition | `llm_reactif_node` | Interaction spontanée via GPT-3.5/4 |
| Passerelle | `gateway_node` | Bridge bi-directionnel ROS 1 ↔ ROS 2 |

---

## Connexion et configuration

### Accès au robot

```bash
# SSH
ssh qtrobot@192.168.100.1

# Interface web (autostart des services)
http://192.168.100.1:8080
http://192.168.100.2:808
```

### Workflow de développement - SSHFS

L'OS du robot est trop ancien pour le Remote SSH de VS Code. On monte le système de fichiers du robot directement en local. Pour cela, crée un dossier `qt_dev` comme suit :

```bash
mkdir ~/qt_dev
```

Puis, pour monter le système de fichiers du robot en local :

```bash
sshfs qtrobot@192.168.100.1:/home/qtrobot/catkin_ws/src ~/qt_dev
```

---

## Installation

### Sur le PC - ROS 2 Humble / Python 3.10

```bash
pip install roslibpy openai==0.28 rclpy opencv-python
```

### Sur le robot - ROS 1 Kinetic / Python 2.7

```bash
# Système
sudo apt-get install python-pyaudio

# Python
pip install "requests<2.28" speechrecognition==3.8.1
```

> **Note vision** : la caméra RealSense et la caméra USB ne peuvent pas tourner simultanément. Désactiver l'une via l'interface web (port 8080) pour éviter les conflits avec Nuitrack.

---

## Mise en route du TCT-DP

### 1. Côté robot - ROS 1

```bash
# Terminal 1 : le bridge de communication
roslaunch rosbridge_server rosbridge_websocket.launch port:=9091

# Terminal 2 : les oreilles (STT Google)
python ~/catkin_ws/src/ears_robot/ears.py
```

### 2. Côté PC - ROS 2

```bash
# Gateway (pont ROS 1 ↔ ROS 2)
ros2 run crearobot_brain gateway

# Orchestrateur de l'expérience
ros2 run crearobot_brain orchestrator

# Interaction : Gère les phases, activation des nodes 
ros2 run crearobot_brain interaction_node

# Cerveau LLM réactif
ros2 run crearobot_brain llm_reactif_node
```

---

## Fonctionnalités clés

### Synchronisation parfaite (lipsync)

La classe `TaskSynchronizer` (basée sur `asyncio`) déclenche simultanément :

- le geste (ROS Service)
- l'expression faciale (ROS Service)
- la parole (TTS)

Un correctif dynamique soustrait le temps de chargement de l'émotion à la durée d'animation de la bouche pour éviter qu'elle ne bouge dans le vide après la fin de la parole.

### Vision & perception

Le `camera_node` récupère le flux vidéo via la Gateway avec deux optimisations :

- **Throttling** : 1 image toutes les 2 à 5 secondes pour ne pas saturer le WiFi
- **Queue management** : `queue_length=1` pour garantir que le LLM analyse toujours l'image la plus récente, sans lag cumulatif

### Intelligence artificielle - LLM

- **Format de sortie** : l'IA répond exclusivement en JSON `["geste", "emotion", "texte"]`
- **Prompt système** : injection des listes fermées `LISTE_GESTES` et `LISTE_EMOTIONS` issues de `config.py` pour donner les gestes et émotions déjà implémentés dans QT.
- **Warm-up** : micro-requête au démarrage pour éliminer le cold start (latence initiale de ~6s réduite à < 2s)

### Micro - réglage AGC (ReSpeaker 4 Mic Array)

```bash
python /home/qtrobot/robot/code/tutorials/examples/voice_activity/tuning.py AGCONOFF 1
python /home/qtrobot/robot/code/tutorials/examples/voice_activity/tuning.py AGCMAXGAIN 100
python /home/qtrobot/robot/code/tutorials/examples/voice_activity/tuning.py AGCDESIREDLEVEL 0.1
```

> Ces réglages sont non persistants et sont réappliqués automatiquement au lancement de `ears.py`.

---

## Tests et démos

| Fichier | Description |
|---|---|
| `demo_interaction.py` | Test simple parole + geste |
| `test_vision.py` | Capture et enregistrement d'une photo `.jpg` |
| `test_gpt.py` | Validation du format JSON et du respect des listes de gestes |
| `test_vision_rt.py` | Affichage du flux vidéo distant en temps réel sur le PC |

---

## Sources

- [Documentation officielle QTrobot](https://docs.luxai.com/docs/intro_code)
- [Wiki ROS QTrobot](https://wiki.ros.org/Robots/qtrobot)
- [Tutoriel Vosk / Python](https://docs.luxai.com/docs/tutorials/python/python_ros_vosk)
- [Documentation micro ReSpeaker](https://docs.luxai.com/docs/v1/modules/microphone)

---

*Développé par Marco G.*
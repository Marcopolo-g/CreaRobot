# Projet Hyppolite — Orchestration TCT-DP & IA Générative

> QTrobot V1 (ROS 1 Kinetic) transformé en compagnon social intelligent. L'intelligence, la vision et l'audition sont déportées sur PC (ROS 2) pour une réactivité maximale.

---

## Architecture du système

Pour garantir une interaction fluide, tous les capteurs (micro, caméra) sont branchés directement sur le PC. La Gateway ne sert plus qu'à envoyer les commandes motrices et vocales au robot.

```
┌────────────────────────────────────────────────────────────────┐
│                          PC (ROS 2)                            │
│  orchestator_node                                              │
|         ↓                                                      │
│ interaction_node → vision_node  ←  [Caméra USB]                |
|                  → stt_node     ←  [Micro USB]                 |     
|                  → projection_node                             | 
|                  → llm_node                                    |
|                                                                |
|                         ↓↑                                     |
│                    gateway_node                                │
└──────────────────────────┬─────────────────────────────────────┘
                           │ WebSocket (rosbridge :9090)
┌──────────────────────────┴─────────────────────────────────────┐
│                   QTRobot (ROS 1)                              │
│                                                                │
│      Moteurs (Gestes)  ·  Écran (Émotions)  ·  TTS (Parole)    │
└────────────────────────────────────────────────────────────────┘
```

| Couche | Node | Rôle |
|---|---|---|
| Logique | `orchestrator_node` | Machine à états — 5 phases TCT-DP |
| Dispatcher | `interaction_node` | Traduit les phases en actions et en gestes / émotions / paroles |
| Audition | `stt_node` | Capture micro local + Google STT |
| Vision | `vision_node` | Flux local via caméra USB externe |
| Cognition | `llm_node` | Chef d'orchestre : analyse le texte, choisit les actions et les images |
| Projection | `projection_node` | Affiche les visuels sur le projecteur HDMI |
| Passerelle | `gateway_node` | Bridge bi-directionnel ROS 1 ↔ ROS 2 |

---

## Connexion et configuration

### Accès au robot

```bash
# SSH
ssh qtrobot@192.168.100.1

# Interface web (autostart des services)
http://192.168.100.1:8080
http://192.168.100.2:8080
```

### Workflow de développement

Pour accéder directement aux fichiers du robot et/ou coder directement dans le robot.

```bash
sftp://qtrobot@192.168.100.1/
```

---

## Installation

### Sur le PC - ROS 2 Humble / Python 3.10

```bash
# Dépendances système
sudo apt install fonts-dejavu fontconfig

# Python
pip install roslibpy openai==0.28 rclpy opencv-python numpy SpeechRecognition PyAudio
```

## Mise en route

### 1. Côté robot - ROS 1

Activer dans l'autostart ros_bridge et motor :
- *start_qt_rosbridge.sh*
- *start_qt_motor.sh*

### 2. Côté PC -ROS 2

```bash
# Orchestrateur de l'expérience
ros2 run crearobot_brain orchestrator

# Interaction : Gère les phases, activation des nodes 
ros2 run crearobot_brain interaction_node

# Cerveau LLM réactif
ros2 run crearobot_brain llm_reactif_node

# La passerelle de commande
ros2 run crearobot_brain gateway

# Le système de projection
ros2 run crearobot_brain projection

# L'audition (micro local)
ros2 run crearobot_brain stt

# La vision (caméra locale)
ros2 run v4l2_camera v4l2_camera_node
```

---

## Fonctionnalités clés

### Synchronisation parfaite (lipsync)

La classe `TaskSynchronizer` (basée sur `asyncio`) déclenche simultanément :

- le geste (ROS Service)
- l'expression faciale (ROS Service)
- la parole (TTS)

Un correctif dynamique soustrait le temps de chargement de l'émotion à la durée d'animation de la bouche pour éviter qu'elle ne bouge dans le vide après la fin de la parole.

### Vision & projection

Le `camera_node` récupère un flux local à 60 FPS sans saturer le WiFi. Le `projection_node` utilise OpenCV pour mapper une fenêtre plein écran sur la sortie HDMI du projecteur, permettant à Hyppolite d'afficher des documents, des décors ou des émotions augmentées.

### Audition déportée - STT

L'utilisation d'un micro cravate via `SpeechRecognition` (Google API) élimine les problèmes de gain du ReSpeaker. Un calcul d'énergie détecte la voix et n'envoie que les segments utiles au cloud, réduisant la latence à moins de 1s.

### Intelligence artificielle — LLM

- **Format de sortie** : l'IA répond exclusivement en JSON `["geste", "emotion", "texte"]`
- **Prompt système** : injection des listes fermées `LISTE_GESTES` et `LISTE_EMOTIONS` issues de `config.py` pour éviter les hallucinations
- **Warm-up** : micro-requête au démarrage pour éliminer le cold start (latence initiale de ~6s réduite à < 2s)

### Verrouillage micro (Half-Duplex)

Le `gateway_node` gère le topic `/pc/is_talking`. Lorsqu'Hyppolite parle, le micro est virtuellement verrouillé pour éviter que le robot ne s'écoute lui-même et ne crée une boucle de rétroaction avec le LLM.

---

## Tests

### 1. Dans le fichier tests

| Fichier | Description |
|---|---|
| `client_bridge.py` | Test de la gateway, communication entre le robot et le pc |
| `test_gpt.py` | Validation du format JSON et du respect des listes de gestes |
| `bridge_client_cam.py` | Affichage du flux vidéo du robot en temps réel sur la tablette du robot |


### 2. Dans le fichier tests_finaux

| Fichier | Description |
|---|---|
| `brain_node_test.py` | Interaction spontanée via GPT-3.5 |
| `camera_node_test.py` | Node de test de la caméra |
| `gateway_node_test.py` | Node de test de la gateway |
| `image_publisher.py` | Envoi d'une image fixe vers le projecteur |

---

## Sources

- [Documentation officielle QTrobot](https://docs.luxai.com/docs/intro_code)
- [Wiki ROS QTrobot](https://wiki.ros.org/Robots/qtrobot)
- [Tutoriel Vosk / Python](https://docs.luxai.com/docs/tutorials/python/python_ros_vosk)
- [Documentation micro ReSpeaker](https://docs.luxai.com/docs/v1/modules/microphone)

---

*Développé par Marco G. — L@b0 Technologie Info-NUMÉRIQUE*

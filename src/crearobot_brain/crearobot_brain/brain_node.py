import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from rclpy.qos import QoSProfile, DurabilityPolicy
import openai
import json
import os
import base64
import io
import subprocess
import tempfile
import glob
import time
import datetime
import numpy as np
import cv2
from sensor_msgs.msg import Image

from . import config


class BrainNode(Node):

    # ── Initialisation ────────────────────────────────────────────────────────

    def __init__(self):
        super().__init__('brain_node')
        self.client = openai.OpenAI(api_key=config.OPENAI_API_KEY)

        self.visual_memory = ""
        self.chat_history  = []
        self.current_phase = ""
        self.current_tour  = 0
        self._c1_ready     = False  # True seulement après la 1ère analyse photo du feedback
        self._c1_robot_turn = 0    # Numéro de prise de parole du robot dans le feedback courant

        self.create_subscription(String, '/pc/vision/image_raw_b64', self.process_vision,  10)
        self.create_subscription(String, '/pc/stt/transcript',        self.handle_dialogue, 10)
        self.create_subscription(String, '/pc/phase_control',         self.phase_callback,  10)

        self.tts_pub   = self.create_publisher(String, '/pc/vision/feedback',  10)
        self.image_pub = self.create_publisher(Image,  '/pc/projector/image',  10)

        self.tctdp_template_b64 = self._load_tctdp_template()
        self.tctdp_template_png_buf = self._precompute_template_png()

        # ── Logger de session ─────────────────────────────────────────────────
        now = datetime.datetime.now()
        day_str  = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H-%M-%S")
        self.session_dir = os.path.join(
            config.SESSIONS_DIR, day_str, time_str
        )
        os.makedirs(self.session_dir, exist_ok=True)
        log_path = os.path.join(self.session_dir, "conversation.log")
        self._log_file = open(log_path, "w", encoding="utf-8")
        self._log_file.write(
            f"=== Session CreaRobot — {day_str} {time_str} — Condition {config.CONDITION} ===\n\n"
        )
        self._log_file.flush()

        # Publie le chemin de session (latched) pour que vision_node utilise le même dossier
        _latched_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._session_dir_pub = self.create_publisher(String, '/pc/session_dir', _latched_qos)
        _sd_msg = String()
        _sd_msg.data = self.session_dir
        self._session_dir_pub.publish(_sd_msg)

        # Abonnement au transcript participant (pour le log)
        self.create_subscription(String, '/pc/stt/transcript',   self._log_participant, 10)
        # Abonnement aux actions robot (pour le log du texte robot)
        self.create_subscription(String, '/pc/qtaction',         self._log_robot,       10)

        self.warmup_llm()
        self.get_logger().info(f"Brain Node prêt — session : {self.session_dir}")

    # ── Gestion des phases ────────────────────────────────────────────────────

    def phase_callback(self, msg):
        data = json.loads(msg.data)
        self.current_phase = data["phase"]
        self.current_tour  = data["tour"]

        if "START_FEEDBACK" in self.current_phase:
            self._c1_ready = False
            self._c1_robot_turn = 0

        if "START_ICE_BREAKING" in self.current_phase:
            if not self.chat_history:
                self.chat_history.append({
                    "role": "assistant",
                    "content": config.ICE_BREAKING_QUESTION
                })

        if "START_TASK_INTRO" in self.current_phase:
            if self.chat_history:
                self.get_logger().info("Réinitialisation mémoire : Ice Breaking effacé.")
                self.chat_history = []

    # ── Analyse visuelle ──────────────────────────────────────────────────────

    def process_vision(self, msg):
        self.get_logger().info("Analyse du dessin en cours...")
        try:
            is_title_phase = "START_TITLE" in self.current_phase

            # C2 feedback : pas besoin de description intermédiaire, on génère l'image directement
            if config.CONDITION == "C2" and not is_title_phase:
                self.generate_c2_feedback(msg.data)
                return

            if is_title_phase:
                prompt = config.PROMPT_TITLE
            elif not self.visual_memory:
                prompt = (
                    "Décris ce que tu vois sur le dessin de maniere a pouvoir te rappeler "
                    "du dessin plus tard sans le voir (entre autre, précise les formes et "
                    "les objets de ce dessin)."
                )
            else:
                prompt = (
                    f"Precedemment, le dessin etait : {self.visual_memory}. "
                    "Dis moi ce qui a change ou ce qui est nouveau ou les modifications "
                    "apportées par rapport à la description précédente. Sois très précis sur les ajouts."
                )

            messages = list(self.chat_history)
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{msg.data}"}}
                ]
            })

            t0 = time.time()
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=300
            )
            self.get_logger().info(f"[LATENCE] GPT vision : {time.time() - t0:.2f} s")
            answer = response.choices[0].message.content

            if is_title_phase:
                self.send_to_robot(answer)
                return

            if not self.visual_memory:
                self.visual_memory = answer
            else:
                self.visual_memory += f"\n[Ajouts récents] : {answer}"

            if config.CONDITION == "C1":
                self.generate_c1_feedback()

        except Exception as e:
            self.get_logger().error(f"Erreur Vision : {e}")

    # ── Feedback C1 (dialogue verbal) ─────────────────────────────────────────

    def generate_c1_feedback(self):
        try:
            self._c1_robot_turn += 1
            turn_instruction = self._c1_turn_instruction()
            messages = [{"role": "system", "content": config.PROMPT_C1}]
            messages.extend(self.chat_history)
            messages.append({"role": "system", "content": f"Dessin actuel : {self.visual_memory}"})
            messages.append({"role": "system", "content": turn_instruction})

            t0 = time.time()
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=100
            )
            self.get_logger().info(f"[LATENCE] GPT C1 feedback : {time.time() - t0:.2f} s")
            text = response.choices[0].message.content
            self.chat_history.append({"role": "assistant", "content": text})
            self._c1_ready = True  # à partir d'ici, handle_dialogue peut répondre
            self.send_to_robot(text)

        except Exception as e:
            self.get_logger().error(f"Erreur C1 : {e}")

    # ── Feedback C2 (génération d'image) ───────────────────────────────

    def _pad_to_square(self, img_cv):
        """ Ajoute des bordures blanches pour obtenir un carre, puis redimensionne en 1024x1024 """
        h, w = img_cv.shape[:2]
        size = max(h, w)
        pad_top    = (size - h) // 2
        pad_bottom = size - h - pad_top
        pad_left   = (size - w) // 2
        pad_right  = size - w - pad_left
        img_sq = cv2.copyMakeBorder(
            img_cv, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
        return cv2.resize(img_sq, (1024, 1024))

    def generate_c2_feedback(self, base64_image):
        try:
            self.get_logger().info("Génération image C2...")

            # ── Décoder et mettre en carré 1024x1024 (requis par gpt-image-1) ──
            img_raw = base64.b64decode(base64_image)
            img_arr = np.frombuffer(img_raw, dtype=np.uint8)
            img_cv  = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            img_sq  = self._pad_to_square(img_cv)
            _, img_png_buf = cv2.imencode('.png', img_sq)

            # ── GPT-4o-mini : description du dessin pour contextualiser gpt-image-1 ──
            t0 = time.time()
            description = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": config.PROMPT_C2_ANALYSIS},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]}
                ],
                max_tokens=350
            ).choices[0].message.content
            self.get_logger().info(f"[LATENCE] GPT-4o-mini description : {time.time() - t0:.2f} s")
            self.get_logger().info(f"Description GPT : {description}")

            edit_prompt = config.PROMPT_C2_EDIT.format(description=description)
            if self.tctdp_template_png_buf is not None:
                images_for_edit = [
                    ("template.png", io.BytesIO(self.tctdp_template_png_buf.tobytes()), "image/png"),
                    ("drawing.png", io.BytesIO(img_png_buf.tobytes()), "image/png"),
                ]
            else:
                images_for_edit = ("drawing.png", io.BytesIO(img_png_buf.tobytes()), "image/png")

            # ── gpt-image-1 edit sans masque ─────────────────────────────────
            # input_fidelity="high" : preserve fidelement les details de l'image d'entree
            # (formes, traits) au lieu de laisser le modele les redessiner/deformer
            t1 = time.time()
            image_response = self.client.images.edit(
                model="gpt-image-1",
                image=images_for_edit,
                prompt=edit_prompt,
                size="1024x1024",
                quality="low",
                input_fidelity="high"
            )
            self.get_logger().info(f"[LATENCE] gpt-image-1 génération : {time.time() - t1:.2f} s")
            self.get_logger().info(f"[LATENCE] C2 total (description + image) : {time.time() - t0:.2f} s")
            img_bytes   = base64.b64decode(image_response.data[0].b64_json)
            img_out_arr = np.frombuffer(img_bytes, dtype=np.uint8)
            img_decoded = cv2.imdecode(img_out_arr, cv2.IMREAD_COLOR)

            # ── Publier vers le projecteur ────────────────────────────────────
            ros_img              = Image()
            ros_img.header.stamp = self.get_clock().now().to_msg()
            ros_img.height, ros_img.width = img_decoded.shape[:2]
            ros_img.encoding     = 'bgr8'
            ros_img.step         = ros_img.width * 3
            ros_img.data         = img_decoded.tobytes()
            self.image_pub.publish(ros_img)
            self.get_logger().info("Image generee et envoyee au projecteur")

            text = "Voila, j'ai essaye quelque chose a partir de ton dessin !"
            self.chat_history.append({"role": "assistant", "content": text})
            self.send_to_robot(text)

        except Exception as e:
            self.get_logger().error(f"Erreur C2 : {e}")


    def handle_dialogue(self, msg):
        is_feedback     = "START_FEEDBACK"    in self.current_phase
        is_ice_breaking = "START_ICE_BREAKING" in self.current_phase

        if not is_feedback and not is_ice_breaking:
            return
        if is_feedback and (not self.visual_memory or not self._c1_ready):
            return

        user_input = msg.data

        if is_ice_breaking:
            messages = [{"role": "system", "content": config.PROMPT_ICE_BREAKING}]
        else:
            self._c1_robot_turn += 1
            turn_instruction = self._c1_turn_instruction()
            messages = [{"role": "system", "content": config.PROMPT_C1}]
            messages.append({"role": "system", "content": f"Memoire visuelle : {self.visual_memory}"})
            messages.append({"role": "system", "content": turn_instruction})

        messages.extend(self.chat_history[-6:])
        messages.append({"role": "user", "content": user_input})

        try:
            t0 = time.time()
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=80
            )
            self.get_logger().info(f"[LATENCE] GPT dialogue : {time.time() - t0:.2f} s")
            answer = response.choices[0].message.content
            self.chat_history.append({"role": "user",      "content": user_input})
            self.chat_history.append({"role": "assistant", "content": answer})
            self.send_to_robot(answer)

        except Exception as e:
            self.get_logger().error(f"Erreur Dialogue ({self.current_phase}) : {e}")

    # ── Utilitaires ───────────────────────────────────────────────────────────

    def _load_tctdp_template(self):
        filename = f"test_sheet_{config.TEST_SHEET_VERSION}.pdf"
        here = os.path.dirname(os.path.abspath(__file__))
        candidats = [
            os.path.join(here, '../../../', filename),
            os.path.join(here, '../../../../../../', filename),
        ]
        pdf_path = next((c for c in candidats if os.path.exists(c)), candidats[0])
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                # 75 DPI suffit pour l'analyse gpt-4o-mini (positions des éléments)
                subprocess.run(
                    ['pdftoppm', '-r', '75', '-l', '1', '-jpeg', pdf_path, f'{tmpdir}/page'],
                    check=True, capture_output=True
                )
                pages = sorted(glob.glob(f'{tmpdir}/page*.jpg'))
                if not pages:
                    raise FileNotFoundError("Aucune page extraite du PDF")
                with open(pages[0], 'rb') as f:
                    b64 = base64.b64encode(f.read()).decode('utf-8')
            self.get_logger().info(f"Template TCT-DP charge ({filename})")
            return b64
        except Exception as e:
            self.get_logger().warning(f"Template TCT-DP non charge : {e}")
            return None

    def _precompute_template_png(self):
        """Pré-calcule le template en PNG 1024x1024 une seule fois au démarrage."""
        if not self.tctdp_template_b64:
            return None
        try:
            tpl_raw = base64.b64decode(self.tctdp_template_b64)
            tpl_arr = np.frombuffer(tpl_raw, dtype=np.uint8)
            tpl_cv  = cv2.imdecode(tpl_arr, cv2.IMREAD_COLOR)
            tpl_sq  = self._pad_to_square(tpl_cv)
            _, buf = cv2.imencode('.png', tpl_sq)
            self.get_logger().info("Template TCT-DP pré-calculé en PNG 1024x1024")
            return buf
        except Exception as e:
            self.get_logger().warning(f"Pré-calcul template PNG échoué : {e}")
            return None

    # ── Logger de conversation ────────────────────────────────────────────────

    def _ts(self):
        return datetime.datetime.now().strftime("%H:%M:%S")

    def _log_participant(self, msg):
        line = f"[{self._ts()}] PARTICIPANT : {msg.data}\n"
        self._log_file.write(line)
        self._log_file.flush()

    def _log_robot(self, msg):
        try:
            data = json.loads(msg.data)
            text = data[2] if len(data) > 2 else ""
        except Exception:
            text = msg.data
        if text and text.strip():
            line = f"[{self._ts()}] ROBOT       : {text}\n"
            self._log_file.write(line)
            self._log_file.flush()

    def _c1_turn_instruction(self):
        if self._c1_robot_turn >= config.MAX_EXCHANGES:
            return "C'est ta DERNIÈRE prise de parole. Conclus en une phrase chaleureuse. INTERDICTION de poser une question."
        return "Tu DOIS terminer ta réponse par une question ouverte et courte."

    def warmup_llm(self):
        try:
            self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1
            )
        except Exception as e:
            self.get_logger().error(f"Echec du warmup gpt-4o-mini : {e}")

        if config.CONDITION == "C2":
            import threading
            threading.Thread(target=self._warmup_image, daemon=True).start()

    def _warmup_image(self):
        try:
            t0 = time.time()
            white = np.full((1024, 1024, 3), 255, dtype=np.uint8)
            _, buf = cv2.imencode('.png', white)
            self.client.images.edit(
                model="gpt-image-1",
                image=("warmup.png", io.BytesIO(buf.tobytes()), "image/png"),
                prompt=".",
                size="1024x1024",
                quality="low",
            )
        except Exception as e:
            self.get_logger().warning(f"Warmup gpt-image-1 échoué : {e}")

    def send_to_robot(self, text):
        msg = String()
        msg.data = text
        self.tts_pub.publish(msg)


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = BrainNode()

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        try:
            node._log_file.close()
        except Exception:
            pass
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()
        os._exit(0)

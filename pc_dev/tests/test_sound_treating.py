import argparse
import sounddevice as sd
from faster_whisper import WhisperModel

# Test direct sans aucune fioriture
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dur", type=float, default=5.0)
    args = parser.parse_args()

    # On charge les deux tailles pour comparer
    print("Chargement des modeles...")
    model_base = WhisperModel("base", device="cpu", compute_type="int8")
    model_small = WhisperModel("small", device="cpu", compute_type="int8")

    input(f"Entree pour enregistrer {args.dur}s (parle normalement)...")
    raw = sd.rec(int(args.dur * 16000), samplerate=16000, channels=1, dtype='float32')
    sd.wait()
    audio = raw.flatten()

    def transcribe(model, label):
        # On utilise le VAD interne qui est tres efficace dans le bruit
        segments, _ = model.transcribe(audio, language="fr", beam_size=5, vad_filter=True)
        text = " ".join(s.text for s in segments).strip()
        print(f"[{label}] : {text}")

    print("\n--- RESULTATS SUR AUDIO BRUT ---")
    transcribe(model_base, "MODELE BASE ")
    transcribe(model_small, "MODELE SMALL")

if __name__ == "__main__":
    main()
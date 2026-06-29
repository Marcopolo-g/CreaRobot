import openai
import json

# Configuration
openai.api_key = "YOUR_API_KEY_HERE"

# --- LISTES DES COMPORTEMENTS DU ROBOT ---
LISTE_GESTES = ["adieu", "angry", "begin", "bored", "bored_long", "breathing_exercise", "bye", "bye-bye", "challenge", "come", "cry", "curious", "ecrit", "fera_mieux", "grandpa", "handclap", "happy", "head_scratch", "hi", "hips", "hug", "kiss", "laugh", "no", "ohno", "point_front", "premiere_recontre", "premiere_rencontre", "protect", "rappel", "sad", "send_kiss", "show_left", "show_QT", "show_right", "show_tablet", "sneezing", "so", "so_what", "stretch", "strong", "surprise", "swipe_left", "swipe_right", "test", "thanks", "up_left", "up_right", "yawn", "yes"] 
LISTE_EMOTIONS = ["afraid", "afraidshort", "angry", "blowing_raspberry", "breathing_exercise", "brushing_teeth", "brushing_teeth_foam", "calmig_down_exercise", "calming_down", "confused", "cry", "dernieradieu", "dirty_face", "dirty_face_sad", "dirty_face_wash", "dirty_teeth", "disgusted", "happy", "happy_blinking", "kiss", "kiss2", "_neutral", "neutral", "_neutral_state_blinking", "neutral_state_blinking", "one_eye_wink", "puffin_the_chredo_eeks", "sad", "scream", "showing_smile", "shy", "surprise", "talking", "talkinglongadapted", "talkinglongrepeat", "very_neutral", "very_neutral_blinking", "very_sad", "with_a_cold", "with_a_cold_cleaning_nose", "with_a_cold_sneezing", "yawn"] 

def appeler_ia_robot(question):
    # On transforme les listes en texte pour le dire à l'IA
    gestes_str = ", ".join(LISTE_GESTES)
    emotions_str = ", ".join(LISTE_EMOTIONS)

    # Le prompt système qui définit les règles
    consigne = f"""
    Tu es un assistant intégré dans un robot social.
    Tes réponses doivent impérativement être une liste Python au format : ["geste", "emotion", "texte"].
    
    RÈGLES STRICTES :
    1. Choisis le geste uniquement parmi : [{gestes_str}].
    2. Choisis l'émotion uniquement parmi : [{emotions_str}].
    3. Le texte doit être en français.
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": consigne},
                {"role": "user", "content": question}
            ]
        )

        resultat = response.choices[0].message.content
        
        # On vérifie que c'est bien du JSON (une liste)
        liste_finale = json.loads(resultat)
        
        print(f"Choix de l'IA :")
        print(f"   - Geste   : {liste_finale[0]}")
        print(f"   - Émotion : {liste_finale[1]}")
        print(f"   - Texte   : {liste_finale[2]}")
        
        return liste_finale

    except Exception as e:
        print(f"Erreur : {e}")
        return None

# Test rapide
appeler_ia_robot("Ma journee se passe trop mal la !")

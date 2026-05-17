import os
import sys
import time
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

MODEL_ID = "openai/whisper-large-v3"
MAX_RETRIES = 3
RETRY_DELAY = 10  # seconds; HF free-tier cold starts typically resolve within 20s


def transcribe(audio_path: str) -> str:
    """Send an audio file to Whisper via HuggingFace InferenceClient and return the Tamil transcript."""
    token = os.getenv("HF_TOKEN")
    if not token:
        raise EnvironmentError("HF_TOKEN not set. Copy .env.example to .env and add your token.")

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    client = InferenceClient(token=token)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Force Tamil — without this Whisper auto-detects and may output Telugu or Malayalam
            result = client.automatic_speech_recognition(audio_path, model=MODEL_ID, language="ta")
            # InferenceClient returns an AutomaticSpeechRecognitionOutput with a .text attribute
            transcript = result.text.strip() if hasattr(result, "text") else str(result).strip()
            if not transcript:
                raise ValueError("Whisper returned an empty transcript.")
            return transcript

        except Exception as e:
            err = str(e)
            if "503" in err and attempt < MAX_RETRIES:
                print(f"[TRANSCRIBE] Model loading (attempt {attempt}/{MAX_RETRIES}), retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
                continue
            raise RuntimeError(f"Whisper transcription failed: {err}") from e

    raise RuntimeError("Whisper model did not load after maximum retries. Try again in a minute.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python transcribe.py <path_to_audio_file>")
        sys.exit(1)

    path = sys.argv[1]
    print(f"[TRANSCRIBE] Sending {path} to Whisper...")
    text = transcribe(path)
    print(f"[TRANSCRIPT] {text}")

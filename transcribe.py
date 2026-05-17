import os
import sys
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

MODEL_ID = "whisper-large-v3"


def transcribe(audio_path: str) -> str:
    """Send an audio file to Whisper via Groq and return the Tamil transcript."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set. Copy .env.example to .env and add your key.")

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    client = Groq(api_key=api_key)

    with open(audio_path, "rb") as audio_file:
        result = client.audio.transcriptions.create(
            file=(os.path.basename(audio_path), audio_file),
            model=MODEL_ID,
            language="ta",
            response_format="text",
        )

    transcript = result.strip() if isinstance(result, str) else result.text.strip()
    if not transcript:
        raise ValueError("Whisper returned an empty transcript.")
    return transcript


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python transcribe.py <path_to_audio_file>")
        sys.exit(1)

    path = sys.argv[1]
    print(f"[TRANSCRIBE] Sending {path} to Whisper via Groq...")
    text = transcribe(path)
    print(f"[TRANSCRIPT] {text}")

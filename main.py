import argparse
import base64
import mimetypes
import os
import time
import datetime
import sys

# Force UTF-8 output so Tamil characters print correctly on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import prompt_builder
import advisor
import logger


def main():
    parser = argparse.ArgumentParser(
        prog="vayal-ai",
        description="Vayal AI — Tamil voice agricultural advisor for Tamil Nadu farmers",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--audio", metavar="PATH", help="Path to a WAV/MP3 audio file with the farmer's question")
    group.add_argument("--text", metavar="QUERY", help="Tamil text question (for testing without audio)")
    group.add_argument("--listen", action="store_true", help="Record 6 seconds from the default microphone")
    parser.add_argument("--image", metavar="PATH", help="Optional photo of the crop or symptom (JPG/PNG)")
    args = parser.parse_args()

    session_start = time.time()
    input_type = "audio" if args.audio else ("listen" if args.listen else "text")
    audio_file = args.audio or None
    image_path = args.image or None
    transcript = None
    thinking = None
    advice = None

    try:
        # --- Step 1: load image (optional) ---
        image_b64 = None
        image_mime = "image/jpeg"
        if args.image:
            if not os.path.exists(args.image):
                raise FileNotFoundError(f"Image file not found: {args.image}")
            image_mime = mimetypes.guess_type(args.image)[0] or "image/jpeg"
            with open(args.image, "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode("utf-8")
            print(f"[IMAGE] Loaded {args.image} ({image_mime})\n")

        # --- Step 2: transcribe ---
        if args.listen:
            import sounddevice as sd
            from scipy.io import wavfile as scipy_wavfile

            sample_rate = 16000  # Whisper expects 16 kHz
            duration = 6
            tmp_wav = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_listen_tmp.wav")

            print("Recording... speak now")
            audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
            sd.wait()
            print("Recording complete\n")

            scipy_wavfile.write(tmp_wav, sample_rate, audio_data)
            try:
                import transcribe as transcribe_mod
                print(f"[STT] Transcribing recorded audio ...")
                t0 = time.time()
                transcript = transcribe_mod.transcribe(tmp_wav)
                elapsed = time.time() - t0
                print(f"[TRANSCRIPT] {transcript}  ({elapsed:.1f}s)\n")
            finally:
                if os.path.exists(tmp_wav):
                    os.remove(tmp_wav)

        elif args.audio:
            import transcribe as transcribe_mod
            print(f"[STT] Transcribing {args.audio} ...")
            t0 = time.time()
            transcript = transcribe_mod.transcribe(args.audio)
            elapsed = time.time() - t0
            print(f"[TRANSCRIPT] {transcript}  ({elapsed:.1f}s)\n")
        else:
            transcript = args.text
            print(f"[QUERY] {transcript}\n")

        # --- Step 3: build prompt ---
        messages = prompt_builder.build(transcript, image_b64=image_b64, image_mime=image_mime)

        # --- Step 4: get advice from Gemma 4 ---
        print("[ADVISOR] Asking Gemma 4 ...")
        t0 = time.time()
        advice, thinking = advisor.get_advice(messages)
        elapsed = time.time() - t0

        if thinking:
            print(f"[THINKING]\n{thinking}\n")

        print(f"\n[VAYAL AI]\n{advice}\n")
        print(f"(responded in {elapsed:.1f}s)")

    except KeyboardInterrupt:
        print("\n[VAYAL AI] Interrupted. Goodbye.")
        sys.exit(0)

    finally:
        # Log whatever we managed to collect, even on partial failure
        if transcript is not None:
            logger.log_session({
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "input_type": input_type,
                "audio_file": audio_file,
                "image_path": image_path,
                "transcript": transcript,
                "thinking": thinking,
                "advice": advice,
                "model": advisor.MODEL_ID,
                "duration_seconds": round(time.time() - session_start, 2),
            })


if __name__ == "__main__":
    main()

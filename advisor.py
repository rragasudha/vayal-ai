import os
import re
import time
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

# gemma-4-26B-A4B-it is the hosted MoE variant: 26B total params, 4B active —
# architecturally identical to the E4B model intended in the project spec.
# E4B-it exists on HF but is not registered as a chat model with any provider.
MODEL_ID = "google/gemma-4-26B-A4B-it"
MAX_RETRIES = 3
RETRY_DELAY = 10  # seconds; same cold-start pattern as transcribe.py

# Matches Gemma 4's thinking block — present when the model reasons before answering.
# We log the thinking for debugging but never show it to the farmer.
_THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _extract_thinking(text: str) -> tuple[str, str | None]:
    """Split response into (clean_advice, thinking_block_or_None)."""
    match = _THINK_PATTERN.search(text)
    thinking = match.group(1).strip() if match else None
    clean = _THINK_PATTERN.sub("", text).strip()
    return clean, thinking


def get_advice(messages: list[dict]) -> tuple[str, str | None]:
    """
    Call Gemma 4 via HuggingFace InferenceClient and return (tamil_advice, thinking).

    The thinking block is extracted and returned separately so main.py can
    log it without displaying it to the farmer.
    """
    token = os.getenv("HF_TOKEN")
    if not token:
        raise EnvironmentError("HF_TOKEN not set. Copy .env.example to .env and add your token.")

    client = InferenceClient(token=token)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_ID,
                messages=messages,
                max_tokens=512,
            )
            raw = response.choices[0].message.content

            if not raw or not raw.strip():
                raise ValueError("Gemma 4 returned an empty response.")

            advice, thinking = _extract_thinking(raw)

            if not advice:
                raise ValueError("No advice content after stripping think block.")

            return advice, thinking

        except Exception as e:
            err = str(e)
            # HF InferenceClient surfaces 503 as an exception with the status code in the message
            if "503" in err and attempt < MAX_RETRIES:
                print(f"[ADVISOR] Model loading (attempt {attempt}/{MAX_RETRIES}), retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
                continue
            raise RuntimeError(f"Gemma 4 inference failed: {err}") from e

    raise RuntimeError("Gemma 4 did not load after maximum retries. Try again in a minute.")


if __name__ == "__main__":
    import prompt_builder

    test_query = "நெல் செடியில் இலைகள் மஞ்சளாக மாறுகின்றன, என்ன செய்வது?"
    print(f"[TEST] Query: {test_query}\n")

    msgs = prompt_builder.build(test_query)
    advice, thinking = get_advice(msgs)

    if thinking:
        print(f"[THINKING]\n{thinking}\n")
    print(f"[ADVICE]\n{advice}")

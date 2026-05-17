import os
import re
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

MODEL_ID = "google/gemma-4-27b-it"

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
    Call Gemma 4 via OpenRouter and return (tamil_advice, thinking).

    The thinking block is extracted and returned separately so main.py can
    log it without displaying it to the farmer.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY not set. Copy .env.example to .env and add your key.")

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )

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


if __name__ == "__main__":
    import prompt_builder

    test_query = "நெல் செடியில் இலைகள் மஞ்சளாக மாறுகின்றன, என்ன செய்வது?"
    print(f"[TEST] Query: {test_query}\n")

    msgs = prompt_builder.build(test_query)
    advice, thinking = get_advice(msgs)

    if thinking:
        print(f"[THINKING]\n{thinking}\n")
    print(f"[ADVICE]\n{advice}")

# Vayal AI — Tamil Voice Agricultural Advisor

Vayal AI is a Tamil-language voice advisory tool for farmers in Tamil Nadu. A farmer speaks a question in Tamil — about crop diseases, pests, planting timing, or soil — and the system responds with practical advice in Tamil. It uses OpenAI Whisper for speech-to-text and Google Gemma 4 for reasoning, both via the HuggingFace Inference API. No database, no UI, no paid services — a CLI script that runs on the HuggingFace free tier.

Built for the [DEV.to Gemma 4 Challenge](https://dev.to/challenges/gemma).

---

## Why Gemma 4

- **Multilingual natively** — understands and responds in Tamil without a translation layer
- **128K+ context window** — the entire agricultural knowledge base fits in one prompt, no RAG needed
- **Thinking mode** — reasons step by step before answering, crucial for distinguishing e.g. nitrogen deficiency from waterlogging
- **Apache 2.0** — NGOs and state governments can deploy freely

The model used is `google/gemma-4-26B-A4B-it` — a Mixture-of-Experts model with 26B total parameters and 4B active parameters per forward pass, giving 4B-class speed at much higher quality.

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/your-username/vayal-ai.git
cd vayal-ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your HuggingFace token
cp .env.example .env
# Edit .env — replace the placeholder with your token from https://huggingface.co/settings/tokens

# 4. Run
python main.py --text "நெல் செடியில் இலைகள் மஞ்சளாக மாறுகின்றன, என்ன செய்வது?"
```

---

## Usage

**Text mode** (for testing and development):
```bash
python main.py --text "நெல் செடியில் இலைகள் மஞ்சளாக மாறுகின்றன, என்ன செய்வது?"
```

**Audio mode** (farmer speaks a question into a microphone):
```bash
python main.py --audio path/to/question.wav
```

---

## Sample output

```
[QUERY] நெல் செடியில் இலைகள் மஞ்சளாக மாறுகின்றன, என்ன செய்வது?

[ADVISOR] Asking Gemma 4 ...

[VAYAL AI]
நிச்சயமாக, உங்கள் நெல் பயிரில் இலைகள் மஞ்சள் நிறமாக மாறுவதற்கு இரண்டு முக்கிய
காரணங்கள் இருக்கலாம். நாம் அதை முதலில் கண்டறிய வேண்டும்.

நைட்ரஜன் குறைபாடு: இலைகள் முதலாவதாகக் கீழே உள்ள பழைய இலைகளில் இருந்து மஞ்சள்
நிறமாக மாறி, மெல்ல மெல்ல மேல் நோக்கி வளர இருந்தால், அது மண்ணில் சத்து குறைபாட்டைக்
குறிக்கிறது.

நீர் தேக்கம்: வயலில் தண்ணீர் அதிகமாகத் தேங்கி நின்றாலோ வேர்கள் அழுகிவிடும்.
செடியை பிடுங்கி பார்த்தால் வேர்கள் கருப்பாக இருக்கும்.

[நைட்ரஜன் குறைபாடு என்றால்]
பஞ்சகாவ்யா கரைசலை 30 மி.லி / லிட்டர் என்ற அளவில் தெளிக்கவும். அல்லது
யூரியாவை 2 கிலோ / 100 லிட்டர் தண்ணீரில் கலந்து தெளிக்கவும்.

[நீர் தேக்கம் என்றால்]
உடனடியாக வயலில் தேங்கிய தண்ணீரை வெளியேற்றவும்.

(responded in 11.2s)
```

---

## HuggingFace free tier notes

- **Cold starts**: The first request after the model has been idle takes 20–60 seconds. The tool retries automatically up to 3 times with 10-second gaps.
- **Rate limits**: The free tier allows a generous number of requests for demo/development use. For sustained production load, consider a paid HF Pro subscription or a dedicated inference endpoint.
- **Inference provider**: Requests are routed through HuggingFace's inference provider network (deepinfra/novita). You pay nothing on the free tier up to monthly credit limits.
- **Whisper STT**: Also uses the HF free tier. Audio files are sent as raw bytes — no local model download needed.
- **Session logs**: Every run is appended to `logs/sessions.jsonl`, including the model's thinking trace (not shown to the farmer but useful for debugging).

---

## Phase 2 vision — fully offline on edge hardware

The current pipeline makes two API calls: Whisper for speech-to-text, then Gemma 4 for reasoning. This requires an internet connection.

Gemma 4 E4B natively supports audio input. Once HuggingFace's Inference API exposes this capability, the entire pipeline collapses into a single model call — spoken Tamil in, Tamil advice out, one round trip. Beyond that, the 4B active-parameter model is small enough to run locally on a Raspberry Pi 5 or a mid-range Android phone using llama.cpp or ExecuTorch. A village health worker or NGO extension officer could carry a phone with the model loaded and answer farmer questions with zero internet dependency. The Apache 2.0 license means any government or NGO can deploy, modify, and distribute the tool freely.

---

## Project structure

```
vayal-ai/
├── main.py              # CLI entry point
├── transcribe.py        # Whisper STT via HF Inference API
├── advisor.py           # Gemma 4 inference via HF Inference API
├── prompt_builder.py    # Assembles system prompt + user query
├── logger.py            # Appends sessions to logs/sessions.jsonl
├── knowledge_base.txt   # Agricultural knowledge for Tamil Nadu crops
├── .env.example         # Token template
└── requirements.txt
```

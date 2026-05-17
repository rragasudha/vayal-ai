# Vayal AI — Architecture

## System Flowchart

```
 python main.py --listen / --audio / --text  [--image]
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│                      main.py                            │
│  argparse parses --listen / --audio / --text / --image  │
└─────────────────────────────────────────────────────────┘
        │
        ├─── --image provided? ──────────────────────────────────────────────┐
        │    base64 + mimetypes encode the file                              │
        │    (stdlib: base64, mimetypes, os)                                 │
        │                                                                    │
        ▼                                                                    │
 ┌──────────────────┐   ┌──────────────────┐   ┌────────────────────────┐   │
 │    --listen       │   │    --audio        │   │       --text           │   │
 │                  │   │                  │   │                        │   │
 │ sounddevice.rec()│   │ pass file path   │   │ use string directly    │   │
 │ records 6s from  │   │ directly to      │   │ skip transcription     │   │
 │ default mic      │   │ transcribe.py    │   │                        │   │
 │                  │   │                  │   └──────────┬─────────────┘   │
 │ scipy.io.wavfile │   └────────┬─────────┘              │                 │
 │ saves _listen_   │            │                         │                 │
 │ tmp.wav locally  │            │                         │                 │
 │                  │            │                         │                 │
 │ deletes tmp after│            │                         │                 │
 └────────┬─────────┘            │                         │                 │
          │                      │                         │                 │
          └──────────────────────┘                         │                 │
                     │                                     │                 │
                     ▼                                     │                 │
          ┌─────────────────────┐                          │                 │
          │    transcribe.py    │                          │                 │
          │                     │                          │                 │
          │ InferenceClient     │ ◄── API CALL 1 ──────►  │                 │
          │ .automatic_speech_  │  HuggingFace routes to   │                 │
          │  recognition()      │  a Whisper provider      │                 │
          │  model: whisper-    │  (deepinfra / novita)    │                 │
          │  large-v3           │  language forced: "ta"   │                 │
          │  language="ta"      │                          │                 │
          └────────┬────────────┘                          │                 │
                   │ Tamil text string                     │                 │
                   └───────────────────────────────────────┘                 │
                                      │                                      │
                                      ▼                                      │
                          ┌───────────────────────┐                          │
                          │   prompt_builder.py   │ ◄────────────────────────┘
                          │                       │  image_b64 (if present)
                          │ Loads knowledge_      │
                          │ base.txt at import    │
                          │                       │
                          │ Builds messages list: │
                          │  [                    │
                          │   {role: system,      │
                          │    content: system_   │
                          │    prompt + full KB}  │
                          │   {role: user,        │
                          │    content: text}     │  ← text only
                          │    OR                 │
                          │    content: [         │  ← multimodal
                          │     {image_url},      │    (with --image)
                          │     {text}            │
                          │    ]}                 │
                          │  ]                    │
                          └──────────┬────────────┘
                                     │ messages list
                                     ▼
                          ┌───────────────────────┐
                          │      advisor.py        │
                          │                       │
                          │ InferenceClient        │ ◄── API CALL 2 ──────►
                          │ .chat.completions      │  HuggingFace routes to
                          │  .create()             │  gemma-4-26B-A4B-it
                          │  model: gemma-4-       │  provider (deepinfra/
                          │  26B-A4B-it            │  novita)
                          │  max_tokens: 512       │
                          │                       │
                          │ strips <think>...</    │
                          │ think> block with re   │
                          │                       │
                          │ returns (advice,       │
                          │          thinking)     │
                          └──────────┬────────────┘
                                     │
                                     ▼
                          ┌───────────────────────┐
                          │      main.py           │
                          │                       │
                          │ prints [VAYAL AI]      │
                          │ advice to terminal     │
                          │                       │
                          │ calls logger.py ──────► logs/sessions.jsonl
                          └───────────────────────┘
```

---

## Libraries and what each does

| Library | Used in | Does what |
|---|---|---|
| `sounddevice` | `main.py` | Talks to the OS microphone driver, captures raw audio samples |
| `scipy.io.wavfile` | `main.py` | Wraps raw samples into a proper WAV file on disk |
| `huggingface_hub.InferenceClient` | `transcribe.py`, `advisor.py` | The single class that handles **both** API calls — Whisper and Gemma 4 |
| `python-dotenv` | `transcribe.py`, `advisor.py` | Reads `HF_TOKEN` from your `.env` file into the environment |
| `re` | `advisor.py` | Strips `<think>...</think>` reasoning blocks from Gemma's raw output |
| `base64`, `mimetypes` | `main.py` | Encodes image files into a data URI for the multimodal message |
| `argparse` | `main.py` | Parses `--listen`, `--audio`, `--text`, `--image` CLI flags |
| `json` | `logger.py` | Serialises each session dict to a JSONL line |

---

## The one token question

You only need **one token — your `HF_TOKEN`**. Here's why:

`InferenceClient` is HuggingFace's own Python client. When you call it, HF's backend looks at the model you requested, finds a provider that hosts it (e.g. deepinfra hosts Whisper, novita hosts Gemma 4), and **proxies the request on your behalf** — billing against your HF account's free credits. You never create a deepinfra or novita account. The routing is invisible.

---

## Where the mic → Whisper code lives

**Step 1 — Recording** (`main.py` lines 53–76):
```python
audio_data = sd.rec(int(6 * 16000), samplerate=16000, channels=1, dtype="int16")
sd.wait()                                    # blocks until 6s are done
scipy_wavfile.write("_listen_tmp.wav", 16000, audio_data)
```

**Step 2 — Sending to Whisper** (`transcribe.py` lines 28–33):
```python
result = client.automatic_speech_recognition(
    audio_path,
    model="openai/whisper-large-v3",
    language="ta"           # forces Tamil script output
)
```

The WAV file is deleted immediately after Whisper returns, whether it succeeds or fails.

---

## API calls summary

| Call # | Where | Method | Model | Provider |
|--------|-------|--------|-------|----------|
| 1 | `transcribe.py` | `InferenceClient.automatic_speech_recognition()` | `openai/whisper-large-v3` | deepinfra / novita |
| 2 | `advisor.py` | `InferenceClient.chat.completions.create()` | `google/gemma-4-26B-A4B-it` | deepinfra / novita |

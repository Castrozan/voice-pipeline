# Voice Pipeline: Real-Time Conversational AI

## Context

The current voice system (hey-bot) records 6s audio chunks → whisper-cpp → OpenClaw Gateway → edge-tts → mpv. ~10s latency, no barge-in, no real conversation. Replace with a real-time, full-duplex voice pipeline (~1-1.5s latency) enabling natural conversations with OpenClaw Claude agents.

## Architecture

```
  Local                              Cloud
┌──────────────────────┐     ┌─────────────────────┐
│ Mic (PipeWire)       │     │ Deepgram Nova-2     │
│   ↓                  │     │ STT ~200ms stream   │
│ VAD (Silero ONNX)    │────▶│       ↓             │
│   ~1ms/frame         │     └──────┬──────────────┘
│   ↓                  │            │ transcript
│ Wake Word Detect     │     ┌──────▼──────────────┐
│   (on transcript)    │     │ Anthropic Claude    │
│                      │     │ via OpenClaw Gateway │
│ OpenClaw Gateway ────│────▶│ (localhost:18790)   │
│   (local daemon)     │     │ SSE ~500ms 1st tok  │
│                      │     └──────┬──────────────┘
│                      │            │ text stream
│ Speaker ◀────────────│◀────┌──────▼──────────────┐
│   ↓                  │     │ OpenAI tts-1        │
│ VAD (barge-in) ──────│────▶│ TTS ~300ms 1st byte │
│                      │     └─────────────────────┘
└──────────────────────┘
```

**Always-on flow:**
1. Mic → VAD → speech detected → open Deepgram WebSocket → stream audio
2. Deepgram returns real-time transcription
3. Scan transcript for wake word (e.g., "jarvis", "robson")
4. Wake word found → continue streaming until Deepgram endpointing signals utterance complete
5. Send post-wake-word text to OpenClaw (Claude)
6. Stream LLM response → stream to OpenAI TTS → play audio
7. After AI responds → **conversation window** (~15s): any speech = follow-up (no wake word)
8. During AI speech → VAD active on echo-cancelled source → user speaks → **barge-in**: cancel LLM + TTS + playback
9. Window expires with no speech → return to ambient wake word listening

**State machine:**
```
AMBIENT → LISTENING → THINKING → SPEAKING → CONVERSING → AMBIENT
   ↑                                 │          │
   └─── timeout (no speech) ────────┘          │
                    ↑                            │
                    └──── barge-in ──────────────┘
```

- **AMBIENT**: VAD + Deepgram running, scanning for wake word
- **LISTENING**: Wake word detected (or in conversation window), collecting utterance via Deepgram endpointing
- **THINKING**: Utterance complete, streaming to OpenClaw
- **SPEAKING**: Playing TTS audio, VAD active for barge-in
- **CONVERSING**: AI finished speaking, conversation window open, any speech triggers LISTENING

## Tech Stack

| Component | Technology | Location |
|-----------|-----------|----------|
| Audio I/O | `sounddevice` (PortAudio) | Local |
| VAD | Silero ONNX (~2MB) | Local |
| STT | Deepgram Nova-2 streaming WS | Cloud |
| LLM | OpenClaw Gateway → Claude | Local daemon → Cloud |
| TTS | OpenAI `tts-1` streaming | Cloud |
| Echo Cancel | PipeWire WebRTC AEC (`echo-cancel-source`) | Local (existing) |
| IPC | Unix domain socket (JSON) | Local |

## Hexagonal Architecture

```
                    ┌─────────────────────────────┐
                    │         DOMAIN CORE          │
                    │                              │
                    │  pipeline.py (state machine) │
                    │  conversation.py (history)   │
                    │  wake_word.py (detection)    │
                    │                              │
                    ├──────────── PORTS ───────────┤
                    │                              │
                    │  AudioPort     (capture/play)│
                    │  VadPort       (speech det)  │
                    │  TranscriberPort (STT)       │
                    │  CompletionPort (LLM)        │
                    │  SynthesizerPort (TTS)       │
                    │  ControlPort   (IPC)         │
                    │                              │
                    └──────────────────────────────┘
                               ▲     ▲
              ┌────────────────┘     └────────────────┐
              │                                       │
    ┌─────────┴──────────┐              ┌─────────────┴─────────┐
    │  DRIVING ADAPTERS   │              │  DRIVEN ADAPTERS       │
    │                     │              │                        │
    │  cli (argparse)     │              │  sounddevice_audio.py  │
    │  control_server.py  │              │  silero_vad.py         │
    │  (unix socket)      │              │  deepgram_stt.py       │
    │                     │              │  openai_whisper_stt.py │
    │                     │              │  openclaw_llm.py       │
    │                     │              │  openai_tts.py         │
    └─────────────────────┘              └────────────────────────┘
```

**Ports** (abstract protocols in domain):
```python
class TranscriberPort(Protocol):
    async def start_session(self) -> None: ...
    async def send_audio(self, frame: bytes) -> None: ...
    async def get_transcript(self) -> AsyncIterator[TranscriptEvent]: ...
    async def close_session(self) -> None: ...

class CompletionPort(Protocol):
    async def stream(self, messages: list[Message], agent: str) -> AsyncIterator[str]: ...
    async def cancel(self) -> None: ...

class SynthesizerPort(Protocol):
    async def synthesize(self, text: str, voice: str) -> AsyncIterator[bytes]: ...
    async def cancel(self) -> None: ...
```

## Project Structure

```
voice-pipeline/                      # Separate repo, flake input
├── flake.nix
├── pyproject.toml
├── nix/
│   ├── package.nix                  # buildPythonApplication
│   └── module.nix                   # Home Manager module
├── src/voice_pipeline/
│   ├── __init__.py
│   ├── __main__.py                  # CLI entry (argparse)
│   │
│   ├── domain/                      # CORE - no external deps
│   │   ├── __init__.py
│   │   ├── pipeline.py              # State machine orchestrator
│   │   ├── conversation.py          # Multi-turn history, barge-in truncation
│   │   ├── wake_word.py             # Keyword detection on transcript text
│   │   ├── state.py                 # State enum + transitions
│   │   └── events.py                # Domain events (SpeechStarted, etc.)
│   │
│   ├── ports/                       # PORTS - interfaces only
│   │   ├── __init__.py
│   │   ├── audio.py                 # AudioPort protocol
│   │   ├── vad.py                   # VadPort protocol
│   │   ├── transcriber.py           # TranscriberPort protocol
│   │   ├── completion.py            # CompletionPort protocol
│   │   ├── synthesizer.py           # SynthesizerPort protocol
│   │   └── control.py               # ControlPort protocol
│   │
│   ├── adapters/                    # ADAPTERS - concrete implementations
│   │   ├── __init__.py
│   │   ├── sounddevice_audio.py     # sounddevice + janus.Queue bridge
│   │   ├── silero_vad.py            # Silero ONNX inference
│   │   ├── deepgram_stt.py          # Deepgram Nova-2 WebSocket
│   │   ├── openai_whisper_stt.py    # OpenAI Whisper batch fallback
│   │   ├── openclaw_llm.py          # httpx SSE to OpenClaw Gateway
│   │   ├── openai_tts.py            # OpenAI tts-1 streaming
│   │   └── unix_control.py          # Unix socket server
│   │
│   └── config.py                    # Pydantic Settings
│
└── tests/
    ├── domain/                      # Pure unit tests, no I/O
    ├── adapters/                    # Integration tests
    └── conftest.py
```

**Python dependencies:**
```toml
[project]
name = "voice-pipeline"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
  "sounddevice>=0.5",
  "numpy>=1.26",
  "onnxruntime>=1.17",
  "httpx>=0.27",
  "httpx-sse>=0.4",
  "openai>=1.30",
  "deepgram-sdk>=3.5",
  "pydantic>=2.7",
  "pydantic-settings>=2.3",
  "janus>=1.0",
]

[project.scripts]
voice-pipeline = "voice_pipeline.__main__:main"
```

## Configuration

```python
class VoicePipelineConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VOICE_PIPELINE_")

    # Gateway
    gateway_url: str = "http://localhost:18790"
    gateway_token_file: str = "/run/agenix/openclaw-gateway-token"
    default_agent: str = "jarvis"

    # STT
    stt_engine: Literal["deepgram", "openai-whisper"] = "deepgram"
    stt_mode: Literal["vad-gated", "always-streaming"] = "vad-gated"
    deepgram_api_key_file: str = ""

    # TTS
    tts_voice: str = "onyx"
    openai_api_key_file: str = ""

    # VAD
    vad_threshold: float = 0.5
    vad_min_silence_ms: int = 300

    # Wake word
    wake_words: list[str] = ["jarvis"]

    # Conversation
    conversation_window_seconds: float = 15.0
    max_history_turns: int = 20

    # Audio
    capture_device: str = "echo-cancel-source"  # PipeWire AEC for speaker echo cancel

    # Barge-in
    barge_in_enabled: bool = True
```

**STT modes (configurable):**
- `vad-gated` (default): Deepgram connection opened only when VAD detects speech. Cheaper.
- `always-streaming`: Continuous Deepgram connection. Lower latency for wake word detection.

## CLI

```bash
voice-pipeline                # Start daemon (always-on)
voice-pipeline --agent robson # Start with specific agent
voice-pipeline toggle         # Toggle on/off via socket
voice-pipeline agent <name>   # Switch agent via socket
voice-pipeline status         # Query state via socket
```

No push-to-talk. Service is on or off.

## Dotfiles Integration

### Files to create/modify

1. **`flake.nix`** — add input:
   ```nix
   voice-pipeline.url = "github:castrozan/voice-pipeline";
   ```

2. **`home/modules/voice-pipeline.nix`** — wraps flake input, derives config from openclaw agents, systemd service

3. **`users/lucas.zanoni/home/voice-pipeline.nix`** — user config:
   ```nix
   services.voice-pipeline = {
     enable = true;
     defaultAgent = "jarvis";
     wakeWords = [ "jarvis" "robson" "jenny" ];
     agents.jarvis.openaiVoice = "onyx";
     agents.robson.openaiVoice = "echo";
     agents.jenny.openaiVoice = "nova";
   };
   ```

4. **`secrets/`** — `deepgram-api-key.age`, `openai-api-key.age`

5. **`.config/hypr/conf.d/bindings.conf`** — toggle keybind:
   ```conf
   bindd = SUPER ALT, V, Voice pipeline toggle, exec, voice-pipeline toggle
   ```

6. **`home/modules/openclaw/config.nix`** — add `tts.openaiVoice` option to agent schema

### Nix packaging (follows voxtype pattern)

The flake exposes `packages.${system}.default` (buildPythonApplication) and `homeManagerModules.default`. Dotfiles imports the HM module and passes the package. Reference: `home/modules/voxtype.nix`.

Fallback if buildPythonApplication is problematic with onnxruntime: use `uv` installation pattern from `home/modules/openclaw/install.nix`.

## Verification

1. **Domain tests**: State machine transitions, conversation truncation, wake word detection — pure Python, no I/O
2. **Adapter tests**: Mock Deepgram/OpenAI responses, verify streaming behavior
3. **Manual test**: `voice-pipeline --agent jarvis`, say "jarvis what time is it", verify response, test barge-in, test follow-up without wake word
4. **Nix build**: `nix build` in voice-pipeline repo, `home-manager switch` in dotfiles
5. **Latency log**: Timestamps at each pipeline stage

## Critical reference files

- `home/modules/hey-bot.nix` — gateway interaction, systemd service, TTS flow
- `home/modules/voxtype.nix` — flake input + HM module pattern
- `home/modules/openclaw/config.nix` — agent schema
- `home/modules/openclaw/install.nix` — fallback uv packaging pattern
- `hosts/dellg15/configs/audio.nix` — PipeWire config
- `.config/hypr/conf.d/bindings.conf` — keybind format

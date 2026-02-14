# Voice Pipeline: Real-Time Conversational AI

Replaces the old hey-bot system (6s audio chunks → whisper-cpp → OpenClaw → edge-tts → mpv, ~10s latency, no barge-in) with a real-time, full-duplex voice pipeline (~1-1.5s latency) enabling natural conversations with OpenClaw Claude agents.

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
│ OpenClaw Gateway ────│────▶│ (localhost:18789)   │
│   (local daemon)     │     │ SSE ~500ms 1st tok  │
│                      │     └──────┬──────────────┘
│                      │            │ text stream
│ Speaker ◀────────────│◀────┌──────▼──────────────┐
│   ↓                  │     │ OpenAI tts-1        │
│ VAD (barge-in) ──────│────▶│ TTS ~300ms 1st byte │
│                      │     └─────────────────────┘
└──────────────────────┘
```

### Always-on Flow

1. Mic → VAD → speech detected → open Deepgram WebSocket → stream audio
2. Deepgram returns real-time transcription
3. Scan transcript for wake word (e.g., "jarvis", "robson")
4. Wake word found → continue streaming until Deepgram endpointing signals utterance complete
5. Send post-wake-word text to OpenClaw (Claude)
6. Stream LLM response → stream to OpenAI TTS → play audio
7. After AI responds → **conversation window** (~15s): any speech = follow-up (no wake word needed)
8. During AI speech → VAD active on echo-cancelled source → user speaks → **barge-in**: cancel LLM + TTS + playback
9. Window expires with no speech → return to ambient wake word listening

No push-to-talk. Service is on or off.

### State Machine

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

### STT Modes

- `vad-gated` (default): Deepgram connection opened only when VAD detects speech. Cheaper.
- `always-streaming`: Continuous Deepgram connection. Lower latency for wake word detection.

## Tech Stack

| Component | Technology | Location |
|-----------|-----------|----------|
| Audio I/O | `sounddevice` (PortAudio) | Local |
| VAD | Silero ONNX v5 (~2MB, 32ms frames) | Local |
| STT | Deepgram Nova-2 streaming WS | Cloud |
| LLM | OpenClaw Gateway → Anthropic Claude | Local daemon → Cloud |
| TTS | OpenAI `tts-1` streaming | Cloud |
| Echo Cancel | PipeWire WebRTC AEC (`echo-cancel-source`) | Local |
| IPC | Unix domain socket (JSON) | Local |

## Hexagonal Architecture

```
                    ┌─────────────────────────────┐
                    │         DOMAIN CORE          │
                    │                              │
                    │  pipeline.py (state machine) │
                    │  speech_detector.py (VAD)    │
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
    │  unix_control.py    │              │  silero_vad.py         │
    │  (unix socket)      │              │  deepgram_stt.py       │
    │                     │              │  openai_whisper_stt.py │
    │                     │              │  openclaw_llm.py       │
    │                     │              │  openai_tts.py         │
    └─────────────────────┘              └────────────────────────┘
```

Ports (abstract protocols in domain):

```python
class TranscriberPort(Protocol):
    async def start_session(self) -> None: ...
    async def send_audio(self, frame: bytes) -> None: ...
    async def get_transcripts(self) -> AsyncIterator[TranscriptEvent]: ...
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
voice-pipeline/
├── flake.nix
├── pyproject.toml
├── nix/
│   ├── package.nix                  # venv + pip install wrapper
│   └── module.nix                   # Home Manager module
├── src/
│   ├── __init__.py
│   ├── __main__.py                  # CLI entry (argparse)
│   ├── config.py                    # Pydantic Settings
│   ├── factory.py                   # Adapter creation, wiring
│   │
│   ├── domain/                      # CORE - no external deps
│   │   ├── __init__.py
│   │   ├── pipeline.py              # State machine orchestrator
│   │   ├── speech_detector.py       # VAD + silence counting, SpeechEvents
│   │   ├── conversation.py          # Multi-turn history, barge-in truncation
│   │   ├── wake_word.py             # Keyword detection on transcript text
│   │   ├── state.py                 # State enum + transitions
│   │   └── events.py                # Domain events
│   │
│   ├── ports/                       # PORTS - interfaces only
│   │   ├── audio.py                 # AudioCapturePort, AudioPlaybackPort
│   │   ├── vad.py                   # VadPort
│   │   ├── transcriber.py           # TranscriberPort
│   │   ├── completion.py            # CompletionPort
│   │   ├── synthesizer.py           # SynthesizerPort
│   │   └── control.py               # ControlPort
│   │
│   └── adapters/                    # ADAPTERS - concrete implementations
│       ├── sounddevice_audio.py     # sounddevice + janus.Queue bridge
│       ├── silero_vad.py            # Silero ONNX inference
│       ├── deepgram_stt.py          # Deepgram Nova-2 WebSocket
│       ├── openai_whisper_stt.py    # OpenAI Whisper batch fallback
│       ├── openclaw_llm.py          # httpx SSE to OpenClaw Gateway
│       ├── openai_tts.py            # OpenAI tts-1 streaming
│       └── unix_control.py          # Unix socket server
│
└── tests/
    ├── conftest.py                  # Shared fixtures and fakes
    ├── domain/                      # Pure unit tests, no I/O
    └── adapters/                    # Integration tests
```

## Usage

```bash
voice-pipeline                  # Start daemon (always-on)
voice-pipeline --agent jarvis   # Start with specific agent
voice-pipeline -v               # Verbose logging
voice-pipeline toggle           # Toggle on/off via socket
voice-pipeline agent <name>     # Switch agent via socket
voice-pipeline status           # Query state via socket
```

## Configuration

All settings via `VOICE_PIPELINE_` prefixed environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `GATEWAY_URL` | `http://localhost:18789` | OpenClaw gateway URL |
| `GATEWAY_TOKEN_FILE` | `/run/agenix/openclaw-gateway-token` | Path to gateway token |
| `DEFAULT_AGENT` | `jarvis` | Default agent name |
| `STT_ENGINE` | `deepgram` | `deepgram` or `openai-whisper` |
| `DEEPGRAM_API_KEY_FILE` | | Path to Deepgram API key |
| `OPENAI_API_KEY_FILE` | | Path to OpenAI API key |
| `TTS_VOICE` | `onyx` | Default TTS voice |
| `MODEL` | `anthropic/claude-sonnet-4-5` | LLM model for OpenClaw |
| `VAD_THRESHOLD` | `0.5` | Speech detection threshold |
| `VAD_MIN_SILENCE_MS` | `300` | Silence duration to end speech |
| `VAD_MODEL_PATH` | | Custom Silero model path |
| `FRAME_DURATION_MS` | `32` | Audio frame size (Silero VAD v5 optimal) |
| `SAMPLE_RATE` | `16000` | Audio sample rate |
| `WAKE_WORDS` | `["jarvis"]` | JSON list of wake words |
| `CAPTURE_DEVICE` | `echo-cancel-source` | PipeWire capture device |
| `BARGE_IN_ENABLED` | `true` | Allow interrupting AI speech |
| `CONVERSATION_WINDOW_SECONDS` | `15.0` | Follow-up window without wake word |
| `MAX_HISTORY_TURNS` | `20` | Max conversation turns kept |
| `AGENT_VOICES` | `{}` | JSON map of agent name to TTS voice |
| `SOCKET_PATH` | `/tmp/voice-pipeline.sock` | Unix control socket path |

## Development

```bash
uv sync
uv pip install pytest pytest-asyncio
uv run pytest tests/ -v
```

## Verification

1. **Domain tests**: State machine transitions, conversation truncation, wake word detection — pure Python, no I/O
2. **Adapter tests**: Mock Deepgram/OpenAI responses, verify streaming behavior
3. **Manual test**: `voice-pipeline --agent jarvis -v`, say "jarvis what time is it", verify response, test barge-in, test follow-up without wake word
4. **Nix build**: `home-manager switch` in dotfiles
5. **Latency log**: Timestamps at each pipeline stage

## Dotfiles Integration

Separate repo, included as a flake input in dotfiles:

```nix
voice-pipeline.url = "github:castrozan/voice-pipeline";
```

Home Manager module at `nix/module.nix` provides `services.voice-pipeline` options and a systemd user service.

User config example:

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

Nix packaging uses venv + pip install pattern (see `nix/package.nix`).

Reference files in dotfiles repo:
- `home/modules/hey-bot.nix` — gateway interaction, systemd service, TTS flow
- `home/modules/voxtype.nix` — flake input + HM module pattern
- `home/modules/openclaw/config.nix` — agent schema
- `home/modules/openclaw/install.nix` — fallback uv packaging pattern
- `hosts/dellg15/configs/audio.nix` — PipeWire config
- `.config/hypr/conf.d/bindings.conf` — keybind format (`SUPER ALT, V` for toggle)

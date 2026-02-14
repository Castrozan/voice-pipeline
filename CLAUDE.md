# Voice Pipeline - Agent Instructions

---
description: Voice pipeline development rules
alwaysApply: true
---

## Project

Real-time voice pipeline: Mic -> VAD -> STT -> Wake Word -> LLM -> TTS -> Speaker.
Hexagonal architecture with ports/adapters pattern. Domain has zero external dependencies.

## Structure

Flat package under `src/`. No `voice_pipeline` namespace — imports are direct:
```python
from config import VoicePipelineConfig
from domain.pipeline import VoicePipeline
from adapters.silero_vad import SileroVad
```

Entry point: `src/__main__.py:main()`
Factory wiring: `src/factory.py:create_pipeline()`

## Key Files

- `src/domain/pipeline.py` — State machine orchestrator, the core loop
- `src/domain/speech_detector.py` — VAD + silence counting, produces SpeechEvents
- `src/factory.py` — Creates all adapters, reads secrets, wires pipeline
- `src/config.py` — Pydantic Settings, all `VOICE_PIPELINE_` env vars
- `nix/module.nix` — Home Manager module for NixOS integration
- `nix/package.nix` — Nix packaging via venv + pip install

## Testing

```bash
uv run pytest tests/ -v
```

All domain tests are pure (no I/O). Fakes in `tests/conftest.py`.
Frame size is 32ms (512 samples at 16kHz) — Silero VAD v5 optimal.

## Conventions

- No comments in code
- No `voice_pipeline.` prefix in imports
- Ports are Protocol classes, adapters are concrete implementations
- Config via environment variables with `VOICE_PIPELINE_` prefix
- State machine: AMBIENT -> LISTENING -> THINKING -> SPEAKING -> CONVERSING

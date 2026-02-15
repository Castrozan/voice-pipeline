import asyncio
import logging
from collections.abc import AsyncIterator

import anthropic

logger = logging.getLogger(__name__)


class AnthropicCompletion:
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-5-20250514") -> None:
        self._client = anthropic.AsyncAnthropic(api_key=api_key or None)
        self._model = model.removeprefix("anthropic/")
        self._cancel_event = asyncio.Event()

    async def stream(
        self,
        messages: list[dict[str, str]],
        agent: str,
    ) -> AsyncIterator[str]:
        self._cancel_event.clear()

        system_prompt = None
        anthropic_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system_prompt = msg["content"]
            else:
                anthropic_messages.append({"role": msg["role"], "content": msg["content"]})

        kwargs: dict = {
            "model": self._model,
            "max_tokens": 1024,
            "messages": anthropic_messages,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        try:
            async with self._client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    if self._cancel_event.is_set():
                        break
                    yield text
        except anthropic.APIStatusError as exc:
            logger.error("Anthropic API error: %s %s", exc.status_code, exc.message)
        except Exception:
            if not self._cancel_event.is_set():
                logger.exception("Anthropic streaming error")

    async def cancel(self) -> None:
        self._cancel_event.set()

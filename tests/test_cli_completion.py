import asyncio
import os

import pytest

from adapters.cli_completion import (
    CliCompletion,
    _format_messages_as_plain_text,
    _build_environment_without_claudecode,
)


class TestFormatMessagesAsPlainText:
    def test_single_user_message(self):
        messages = [{"role": "user", "content": "Hello"}]
        result = _format_messages_as_plain_text(messages)
        assert result == "User: Hello"

    def test_system_and_user_messages(self):
        messages = [
            {"role": "system", "content": "You are a voice assistant."},
            {"role": "user", "content": "What time is it?"},
        ]
        result = _format_messages_as_plain_text(messages)
        assert "System: You are a voice assistant." in result
        assert "User: What time is it?" in result

    def test_multi_turn_conversation(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How are you?"},
        ]
        result = _format_messages_as_plain_text(messages)
        parts = result.split("\n\n")
        assert len(parts) == 4
        assert parts[0] == "System: You are helpful."
        assert parts[1] == "User: Hi"
        assert parts[2] == "Assistant: Hello!"
        assert parts[3] == "User: How are you?"

    def test_empty_messages(self):
        result = _format_messages_as_plain_text([])
        assert result == ""


class TestBuildEnvironmentWithoutClaudecode:
    def test_removes_claudecode_variable(self, monkeypatch):
        monkeypatch.setenv("CLAUDECODE", "some-value")
        environment = _build_environment_without_claudecode()
        assert "CLAUDECODE" not in environment

    def test_preserves_other_variables(self, monkeypatch):
        monkeypatch.setenv("MY_TEST_VAR", "kept")
        environment = _build_environment_without_claudecode()
        assert environment["MY_TEST_VAR"] == "kept"

    def test_works_when_claudecode_not_set(self, monkeypatch):
        monkeypatch.delenv("CLAUDECODE", raising=False)
        environment = _build_environment_without_claudecode()
        assert "CLAUDECODE" not in environment


class TestCliCompletionStream:
    @pytest.mark.asyncio
    async def test_streams_stdout_from_echo_command(self):
        completion = CliCompletion(command="cat")
        messages = [{"role": "user", "content": "hello world"}]

        chunks = []
        async for chunk in completion.stream(messages, agent="test"):
            chunks.append(chunk)

        full_output = "".join(chunks)
        assert "User: hello world" in full_output

    @pytest.mark.asyncio
    async def test_handles_failing_command(self):
        completion = CliCompletion(command="false")
        messages = [{"role": "user", "content": "test"}]

        chunks = []
        async for chunk in completion.stream(messages, agent="test"):
            chunks.append(chunk)

        assert chunks == []

    @pytest.mark.asyncio
    async def test_cancel_kills_process(self):
        completion = CliCompletion(command="sleep 60")
        messages = [{"role": "user", "content": "test"}]

        async def cancel_after_short_delay():
            await asyncio.sleep(0.1)
            await completion.cancel()

        cancel_task = asyncio.create_task(cancel_after_short_delay())

        chunks = []
        async for chunk in completion.stream(messages, agent="test"):
            chunks.append(chunk)

        await cancel_task

    @pytest.mark.asyncio
    async def test_uses_agent_parameter_in_messages(self):
        completion = CliCompletion(command="cat")
        messages = [
            {"role": "system", "content": "You are jarvis."},
            {"role": "user", "content": "speak"},
        ]

        chunks = []
        async for chunk in completion.stream(messages, agent="jarvis"):
            chunks.append(chunk)

        full_output = "".join(chunks)
        assert "System: You are jarvis." in full_output

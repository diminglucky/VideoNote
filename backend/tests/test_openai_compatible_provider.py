from types import SimpleNamespace

import pytest

from app.agents.agent_trace import JsonlTraceStore
from app.agents.llm_agents import LlmAgentClient
from app.gpt.provider import OpenAI_compatible_provider as provider_module
from app.gpt.provider.OpenAI_compatible_provider import (
    OpenAICompatibleProvider,
    normalize_openai_base_url,
)


def test_normalize_openai_base_url_adds_v1_only_for_root_paths():
    assert normalize_openai_base_url("https://api.spinortech.ai/") == "https://api.spinortech.ai/v1"
    assert normalize_openai_base_url("https://api.deepseek.com") == "https://api.deepseek.com/v1"
    assert (
        normalize_openai_base_url("https://generativelanguage.googleapis.com/v1beta/openai/")
        == "https://generativelanguage.googleapis.com/v1beta/openai"
    )


def test_connection_rejects_non_completion_response(monkeypatch):
    class _Completions:
        @staticmethod
        def create(**_kwargs):
            return "<!doctype html><html>gateway page</html>"

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=_Completions()),
    )
    monkeypatch.setattr(
        provider_module,
        "build_openai_client",
        lambda *_args, **_kwargs: fake_client,
    )

    assert OpenAICompatibleProvider.test_connection(
        api_key="test-key",
        base_url="https://api.spinortech.ai/",
        model="deepseek-v4-flash",
    ) is False


def test_connection_accepts_valid_chat_completion(monkeypatch):
    class _Completions:
        @staticmethod
        def create(**_kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
            )

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=_Completions()),
    )
    monkeypatch.setattr(
        provider_module,
        "build_openai_client",
        lambda *_args, **_kwargs: fake_client,
    )

    assert OpenAICompatibleProvider.test_connection(
        api_key="test-key",
        base_url="https://api.spinortech.ai/",
        model="deepseek-v4-flash",
    ) is True


def test_llm_agent_client_rejects_non_completion_response(tmp_path):
    class _Completions:
        @staticmethod
        def create(**_kwargs):
            return "<!doctype html><html>gateway page</html>"

    gpt = SimpleNamespace(
        model="deepseek-v4-flash",
        client=SimpleNamespace(chat=SimpleNamespace(completions=_Completions())),
    )
    client = LlmAgentClient(gpt, JsonlTraceStore(tmp_path / "trace.jsonl"))

    with pytest.raises(RuntimeError, match="非 ChatCompletion"):
        client.complete(
            "supervisor",
            [{"role": "user", "content": "ping"}],
            task_id="task-1",
        )

import logging
import os
from unittest.mock import patch

import pytest
from litellm.exceptions import NotFoundError, RateLimitError

from mesa_llm.module_llm import ModuleLLM


class TestModuleLLM:
    """Test ModuleLLM class"""

    @pytest.fixture(autouse=True)
    def _mock_model_catalog_checks(self, monkeypatch):
        """Keep model validation deterministic in unit tests."""

        monkeypatch.setattr(
            "mesa_llm.module_llm.litellm.get_model_info", lambda model: {}
        )

    def test_missing_provider_prefix(self):
        """ModuleLLM should raise ValueError when llm_model has no provider prefix."""
        with pytest.raises(ValueError, match="Invalid model format"):
            ModuleLLM(llm_model="gpt-4o")

    def test_module_llm_initialization(self, mock_environment):
        # Test initialization with default values
        llm = ModuleLLM(llm_model="openai/gpt-4o")
        assert llm.api_key == "test_openai_key"
        assert llm.api_base is None
        assert llm.llm_model == "openai/gpt-4o"
        assert llm.system_prompt is None

        # Test initialization with ollama provider
        llm = ModuleLLM(llm_model="ollama/llama2")
        assert llm.api_base == "http://localhost:11434"
        assert llm.llm_model == "ollama/llama2"
        assert llm.system_prompt is None

        # Test initialization with ollama provider + custom api_base
        llm = ModuleLLM(llm_model="ollama/llama2", api_base="http://localhost:99999")
        assert llm.api_base == "http://localhost:99999"
        assert llm.llm_model == "ollama/llama2"
        assert llm.system_prompt is None

        # Test init without api_key in dotenv
        with patch.dict(os.environ, {}, clear=True), pytest.raises(ValueError):
            ModuleLLM(llm_model="openai/gpt-4o")

    def test_build_messages(self):
        # Test _build_messages with string prompt
        llm = ModuleLLM(llm_model="openai/gpt-4o")
        messages = llm._build_messages("Hello, how are you?")
        assert messages == [
            {"role": "system", "content": ""},
            {"role": "user", "content": "Hello, how are you?"},
        ]

        # Test _build_messages with list of prompts
        messages = llm._build_messages(
            ["Hello, how are you?", "What is the weather in Tokyo?"]
        )
        assert messages == [
            {"role": "system", "content": ""},
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "user", "content": "What is the weather in Tokyo?"},
        ]

        # Test _build_messages with system prompt
        llm = ModuleLLM(
            llm_model="openai/gpt-4o", system_prompt="You are a helpful assistant."
        )
        messages = llm._build_messages("Hello, how are you?")
        assert messages == [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
        ]

        # Test _build_messages with system prompt and list of prompts
        messages = llm._build_messages(
            ["Hello, how are you?", "What is the weather in Tokyo?"]
        )
        assert messages == [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "user", "content": "What is the weather in Tokyo?"},
        ]

        # Test _build_messages no system prompt and no prompt
        llm = ModuleLLM(llm_model="openai/gpt-4o")
        messages = llm._build_messages(prompt=None)
        assert messages == [{"role": "system", "content": ""}]

        # Test _build_messages with invalid prompt type raises TypeError
        with pytest.raises(TypeError, match="Invalid prompt type"):
            llm._build_messages(prompt=123)

        # Test _build_messages with non-string list element raises TypeError
        with pytest.raises(TypeError, match="Invalid prompt list element"):
            llm._build_messages(prompt=["valid", 123])

    def test_model_name_validity(self):
        # Test initialization with invalid model format

        with pytest.raises(ValueError, match="Invalid model format"):
            ModuleLLM(llm_model="gpt-4o")

        llm = ModuleLLM(llm_model="openai/gpt-4o")
        assert llm.llm_model == "openai/gpt-4o"

        llm = ModuleLLM(llm_model="ollama/llama2")
        assert llm.llm_model == "ollama/llama2"

    def test_custom_api_base_allows_unmapped_model(self, monkeypatch, caplog):
        def _raise_unmapped(model):
            raise Exception("This model isn't mapped yet")

        monkeypatch.setattr(
            "mesa_llm.module_llm.litellm.get_model_info", _raise_unmapped
        )
        monkeypatch.setattr(
            "mesa_llm.module_llm.litellm.supports_function_calling",
            lambda model: pytest.fail(
                "supports_function_calling should not run for unmapped models"
            ),
        )

        with caplog.at_level(logging.WARNING):
            llm = ModuleLLM(
                llm_model="openai/my-local-model",
                api_base="http://localhost:1234/v1",
            )

        assert llm.llm_model == "openai/my-local-model"
        assert llm.api_base == "http://localhost:1234/v1"
        assert "does not support function calling" not in caplog.text

    def test_generate_invalid_model_name_has_descriptive_error(self, monkeypatch):
        monkeypatch.setattr(
            "mesa_llm.module_llm.completion",
            lambda **kwargs: (_ for _ in ()).throw(
                NotFoundError(
                    message="OpenAIException - model_not_found",
                    model="openai/not-a-real-model",
                    llm_provider="openai",
                )
            ),
        )

        llm = ModuleLLM(llm_model="openai/not-a-real-model")

        with pytest.raises(ValueError, match="Invalid or unsupported model"):
            ModuleLLM.generate.__wrapped__(llm, prompt="Hello, how are you?")

    def test_mapped_model_without_function_calling_warns(self, monkeypatch, caplog):
        monkeypatch.setattr(
            "mesa_llm.module_llm.litellm.get_model_info", lambda model: {}
        )
        monkeypatch.setattr(
            "mesa_llm.module_llm.litellm.supports_function_calling",
            lambda model: False,
        )

        with caplog.at_level(logging.WARNING):
            ModuleLLM(llm_model="openai/gpt-4o")

        assert "does not support function calling" in caplog.text

    def test_generate(self, monkeypatch, llm_response_factory):
        monkeypatch.setattr(
            "mesa_llm.module_llm.completion", lambda **kwargs: llm_response_factory()
        )

        # Test generate with string prompt
        llm = ModuleLLM(llm_model="openai/gpt-4o")
        response = llm.generate(prompt="Hello, how are you?")
        assert response is not None

        # Test generate with list of prompts
        response = llm.generate(
            prompt=["Hello, how are you?", "What is the weather in Tokyo?"]
        )
        assert response is not None

        # Test generate with string prompt for Ollama
        llm = ModuleLLM(llm_model="ollama/llama2")
        response = llm.generate(prompt="Hello, how are you?")
        assert response is not None

        # Test generate with list of prompts
        response = llm.generate(
            prompt=["Hello, how are you?", "What is the weather in Tokyo?"]
        )
        assert response is not None

    def test_generate_rewrites_rate_limit_error_with_openai_docs(self, monkeypatch):
        original_error = RateLimitError(
            "per-minute limit hit", "openai", "openai/gpt-4o"
        )

        def _raise_rate_limit(**kwargs):
            raise original_error

        monkeypatch.setattr("mesa_llm.module_llm.completion", _raise_rate_limit)

        llm = ModuleLLM(llm_model="openai/gpt-4o")
        with pytest.raises(RateLimitError) as exc_info:
            ModuleLLM.generate.__wrapped__(llm, prompt="Hello, how are you?")

        assert str(exc_info.value) == (
            "litellm.RateLimitError: Rate limit exceeded for model "
            "'openai/gpt-4o'. per-minute limit hit Please wait a few minutes and "
            "try again, or switch to a different model. To check your quota visit: "
            "https://developers.openai.com/api/docs/guides/rate-limits"
        )

    def test_generate_rewrites_rate_limit_error_with_gemini_docs(self, monkeypatch):
        original_error = RateLimitError(
            'geminiException - {"error": {"code": 429}}',
            "gemini",
            "gemini/gemini-2.0-flash",
            max_retries=5,
            num_retries=3,
        )

        def _raise_rate_limit(**kwargs):
            raise original_error

        monkeypatch.setattr("mesa_llm.module_llm.completion", _raise_rate_limit)

        llm = ModuleLLM(llm_model="gemini/gemini-2.0-flash")
        with pytest.raises(RateLimitError) as exc_info:
            ModuleLLM.generate.__wrapped__(llm, prompt="Hello, how are you?")

        assert str(exc_info.value) == (
            "litellm.RateLimitError: Rate limit exceeded for model "
            '\'gemini/gemini-2.0-flash\'. geminiException - {"error": {"code": '
            "429}} Please wait a few minutes and try again, or switch to a "
            "different model. To check your quota visit: "
            "https://ai.google.dev/gemini-api/docs/rate-limits LiteLLM Retried: "
            "3 times, LiteLLM Max Retries: 5"
        )

    @pytest.mark.asyncio
    async def test_agenerate(self, monkeypatch, llm_response_factory):
        async def _dummy_acompletion(**kwargs):
            return llm_response_factory()

        monkeypatch.setattr("mesa_llm.module_llm.acompletion", _dummy_acompletion)
        # Test agenerate with string prompt
        llm = ModuleLLM(llm_model="openai/gpt-4o")
        response = await llm.agenerate(prompt="Hello, how are you?")
        assert response is not None

        # Test agenerate with list of prompts
        response = await llm.agenerate(
            prompt=["Hello, how are you?", "What is the weather in Tokyo?"]
        )
        assert response is not None

    @pytest.mark.asyncio
    async def test_agenerate_rewrites_rate_limit_error_with_openrouter_docs(
        self, monkeypatch
    ):
        class _SingleAttempt:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        class _SingleAsyncRetrying:
            def __init__(self, **kwargs):
                self._yielded = False

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self._yielded:
                    raise StopAsyncIteration
                self._yielded = True
                return _SingleAttempt()

        async def _raise_rate_limit(**kwargs):
            raise RateLimitError(
                "provider throttle triggered",
                "openrouter",
                "openrouter/openai/gpt-4o",
            )

        monkeypatch.setenv("OPENROUTER_API_KEY", "test_openrouter_key")
        monkeypatch.setattr("mesa_llm.module_llm.AsyncRetrying", _SingleAsyncRetrying)
        monkeypatch.setattr("mesa_llm.module_llm.acompletion", _raise_rate_limit)

        llm = ModuleLLM(llm_model="openrouter/openai/gpt-4o")
        with pytest.raises(RateLimitError) as exc_info:
            await llm.agenerate(prompt="Hello, how are you?")

        assert str(exc_info.value) == (
            "litellm.RateLimitError: Rate limit exceeded for model "
            "'openrouter/openai/gpt-4o'. provider throttle triggered Please wait a "
            "few minutes and try again, or switch to a different model. To check "
            "your quota visit: https://openrouter.ai/docs/api/reference/limits"
        )

    def test_generate_accepts_per_call_system_prompt(
        self, monkeypatch, llm_response_factory
    ):
        captured = {}

        def _dummy_completion(**kwargs):
            captured.update(kwargs)
            return llm_response_factory()

        monkeypatch.setattr("mesa_llm.module_llm.completion", _dummy_completion)

        llm = ModuleLLM(llm_model="openai/gpt-4o", system_prompt="base-system-prompt")
        llm.generate(prompt="Hello", system_prompt="scoped-system-prompt")

        assert captured["messages"][0]["content"] == "scoped-system-prompt"
        assert llm.system_prompt == "base-system-prompt"

    def test_generate_can_suppress_hidden_reasoning_without_overriding_core_options(
        self, monkeypatch, llm_response_factory
    ):
        captured = {}

        def _dummy_completion(**kwargs):
            captured.update(kwargs)
            return llm_response_factory()

        monkeypatch.setattr("mesa_llm.module_llm.completion", _dummy_completion)

        llm = ModuleLLM(llm_model="openai/gpt-4o")
        tool_schema = [{"type": "function", "function": {"name": "read_state"}}]
        response_format = {"type": "json_object"}

        llm.generate(
            prompt="Hello",
            tool_schema=tool_schema,
            tool_choice="none",
            response_format=response_format,
            suppress_thinking=True,
        )

        assert captured["think"] is False
        assert captured["drop_params"] is True
        assert captured["model"] == "openai/gpt-4o"
        assert captured["messages"] == [
            {"role": "system", "content": ""},
            {"role": "user", "content": "Hello"},
        ]
        assert captured["tools"] is tool_schema
        assert captured["tool_choice"] == "none"
        assert captured["response_format"] is response_format

    @pytest.mark.asyncio
    async def test_agenerate_accepts_per_call_system_prompt(
        self, monkeypatch, llm_response_factory
    ):
        captured = {}

        async def _dummy_acompletion(**kwargs):
            captured.update(kwargs)
            return llm_response_factory()

        monkeypatch.setattr("mesa_llm.module_llm.acompletion", _dummy_acompletion)

        llm = ModuleLLM(llm_model="openai/gpt-4o", system_prompt="base-system-prompt")
        await llm.agenerate(prompt="Hello", system_prompt="scoped-system-prompt")

        assert captured["messages"][0]["content"] == "scoped-system-prompt"
        assert llm.system_prompt == "base-system-prompt"

    @pytest.mark.asyncio
    async def test_agenerate_can_suppress_hidden_reasoning_without_overriding_core_options(
        self, monkeypatch, llm_response_factory
    ):
        captured = {}

        async def _dummy_acompletion(**kwargs):
            captured.update(kwargs)
            return llm_response_factory()

        monkeypatch.setattr("mesa_llm.module_llm.acompletion", _dummy_acompletion)

        llm = ModuleLLM(llm_model="openai/gpt-4o")
        tool_schema = [{"type": "function", "function": {"name": "read_state"}}]
        response_format = {"type": "json_object"}

        await llm.agenerate(
            prompt="Hello",
            tool_schema=tool_schema,
            tool_choice="none",
            response_format=response_format,
            suppress_thinking=True,
        )

        assert captured["think"] is False
        assert captured["drop_params"] is True
        assert captured["model"] == "openai/gpt-4o"
        assert captured["messages"] == [
            {"role": "system", "content": ""},
            {"role": "user", "content": "Hello"},
        ]
        assert captured["tools"] is tool_schema
        assert captured["tool_choice"] == "none"
        assert captured["response_format"] is response_format

    @pytest.mark.asyncio
    async def test_agenerate_invalid_model_name_has_descriptive_error(
        self, monkeypatch
    ):
        class _SingleAttempt:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        class _SingleAsyncRetrying:
            def __init__(self, **kwargs):
                self._yielded = False

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self._yielded:
                    raise StopAsyncIteration
                self._yielded = True
                return _SingleAttempt()

        async def _raise_invalid_model(**kwargs):
            raise Exception(
                "This model isn't mapped yet. model=openai/not-a-real-model, "
                "custom_llm_provider=openai. Add it here - "
                "https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json."
            )

        monkeypatch.setattr("mesa_llm.module_llm.AsyncRetrying", _SingleAsyncRetrying)
        monkeypatch.setattr("mesa_llm.module_llm.acompletion", _raise_invalid_model)

        llm = ModuleLLM(llm_model="openai/not-a-real-model")

        with pytest.raises(ValueError, match="Invalid or unsupported model"):
            await llm.agenerate(prompt="Hello, how are you?")

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from dotenv import load_dotenv
from litellm import acompletion, completion, litellm
from litellm.exceptions import (
    APIConnectionError,
    NotFoundError,
    RateLimitError,
    Timeout,
)
from tenacity import AsyncRetrying, retry, retry_if_exception, wait_exponential

RETRYABLE_EXCEPTIONS = (
    APIConnectionError,
    Timeout,
    RateLimitError,
)
_ONE_SHOT_COMPLETION = ContextVar("mesa_llm_one_shot_completion", default=False)


def _should_retry_completion(error: BaseException) -> bool:
    return not _ONE_SHOT_COMPLETION.get() and isinstance(error, RETRYABLE_EXCEPTIONS)


load_dotenv()
logger = logging.getLogger(__name__)


class ModuleLLM:
    """
    A module that provides a simple interface for using LLMs

    Note : Currently supports OpenAI, Anthropic, xAI, Huggingface, Ollama, OpenRouter, NovitaAI, Gemini
    """

    def __init__(
        self,
        llm_model: str,
        fallback_model: str | None = None,
        api_base: str | None = None,
        system_prompt: str | None = None,
    ):
        """
        Initialize the LLM module

        Args:
            llm_model: The model to use for the LLM in the format
                "{provider}/{model}" (for example, "openai/gpt-4o").
            api_base: The API base to use if the LLM provider is Ollama
            system_prompt: The system prompt to use for the LLM

        Raises:
            ValueError: If llm_model is not in the expected "{provider}/{model}"
                format, or if the provider API key is missing.
        """
        self.api_base = api_base
        self.llm_model = llm_model
        self.fallback_model = fallback_model
        self.system_prompt = system_prompt
        self._fallback_active = False

        if "/" not in llm_model:
            raise ValueError(
                f"Invalid model format '{llm_model}'. "
                "Expected '{provider}/{model}', e.g. 'openai/gpt-4o'."
            )

        provider = self.llm_model.split("/")[0].upper()

        if provider in ["OLLAMA", "OLLAMA_CHAT"]:
            if self.api_base is None:
                self.api_base = "http://localhost:11434"
                logger.warning(
                    "Using default Ollama API base: %s. If inference is not working, you may need to set the API base to the correct URL.",
                    self.api_base,
                )
        else:
            try:
                self.api_key = os.environ[f"{provider}_API_KEY"]
            except KeyError as err:
                if self.fallback_model is not None:
                    logger.warning(
                        "Primary provider '%s' API key is missing; fallback model '%s' will be used when the primary call is attempted.",
                        provider,
                        self.fallback_model,
                    )
                    self.api_key = None
                else:
                    raise ValueError(
                        f"No API key found for {provider}. Please set the {provider}_API_KEY environment variable (e.g., in your .env file)."
                    ) from err

        try:
            litellm.get_model_info(model=self.llm_model)
        except Exception as exc:
            logger.debug(
                "Skipping function-calling capability check for unmapped model %s: %s",
                self.llm_model,
                exc,
            )
        else:
            if not litellm.supports_function_calling(model=self.llm_model):
                logger.warning(
                    "%s does not support function calling. This model may not be able to use tools. Please check the model documentation at https://docs.litellm.ai/docs/providers for more information.",
                    self.llm_model,
                )

    @property
    def active_model(self) -> str:
        if self._fallback_active and self.fallback_model:
            return self.fallback_model
        return self.llm_model

    def _switch_to_fallback(self):
        if self.fallback_model and not self._fallback_active:
            self._fallback_active = True
            logger.warning(
                "Primary model '%s' failed; switching permanently to fallback model '%s'.",
                self.llm_model,
                self.fallback_model,
            )

    def _build_invalid_model_error(self, error: Exception) -> ValueError:
        provider = self.llm_model.split("/", 1)[0].lower()
        return ValueError(
            f"Invalid or unsupported model '{self.llm_model}' for provider "
            f"'{provider}'. Details: {error}. "
            "Please verify the model name and provider prefix, and if you are "
            f"using a custom endpoint set the correct api_base (current: {self.api_base})."
        )

    def _build_messages(
        self,
        prompt: str | list[str] | None = None,
        system_prompt: str | None = None,
    ) -> list[dict]:
        """
        Format the prompt messages for the LLM of the form : {"role": ..., "content": ...}

        Args:
            prompt: The prompt to generate a response for (str, list of strings, or None)
            system_prompt: Optional system prompt scoped to this call only.

        Returns:
            The messages for the LLM
        """
        messages = []

        # Always include a system message. Default to empty string if no system
        # prompt to support Ollama.
        system_content = (
            self.system_prompt if system_prompt is None else system_prompt
        ) or ""
        messages.append({"role": "system", "content": system_content})

        if prompt is not None:
            if isinstance(prompt, str):
                messages.append({"role": "user", "content": prompt})
            elif isinstance(prompt, list):
                invalid_prompt = next(
                    (
                        (index, value)
                        for index, value in enumerate(prompt)
                        if not isinstance(value, str)
                    ),
                    None,
                )
                if invalid_prompt is not None:
                    index, value = invalid_prompt
                    raise TypeError(
                        f"Invalid prompt list element at index {index}: "
                        f"type '{type(value).__name__}'. Expected str."
                    )
                messages.extend([{"role": "user", "content": p} for p in prompt])
            else:
                raise TypeError(
                    f"Invalid prompt type '{type(prompt).__name__}'. "
                    "Expected str, list[str], or None."
                )
        return messages

    def _build_rate_limit_error(self, error: RateLimitError) -> RateLimitError:
        provider = self.llm_model.split("/", 1)[0].lower()
        docs_url = {
            "anthropic": "https://platform.claude.com/docs/en/api/rate-limits",
            "gemini": "https://ai.google.dev/gemini-api/docs/rate-limits",
            "novita": "https://novita.ai/docs/guides/llm-rate-limits",
            "openai": "https://developers.openai.com/api/docs/guides/rate-limits",
            "openrouter": "https://openrouter.ai/docs/api/reference/limits",
            "xai": "https://docs.x.ai/developers/rate-limits",
        }.get(provider)

        detail = error.message.removeprefix("litellm.RateLimitError: ").strip()
        message_parts = [f"Rate limit exceeded for model '{self.llm_model}'."]
        if detail:
            message_parts.append(detail)
        message_parts.append(
            "Please wait a few minutes and try again, or switch to a different model."
        )
        if docs_url:
            message_parts.append(f"To check your quota visit: {docs_url}")

        message = " ".join(message_parts)
        return RateLimitError(
            message=message,
            llm_provider=error.llm_provider,
            model=error.model,
            response=error.response,
            litellm_debug_info=error.litellm_debug_info,
            max_retries=error.max_retries,
            num_retries=error.num_retries,
        )

    @contextmanager
    def _one_shot_completion(self) -> Iterator[None]:
        """Disable retry and fallback layers for one completion request."""
        token = _ONE_SHOT_COMPLETION.set(True)
        try:
            yield
        finally:
            _ONE_SHOT_COMPLETION.reset(token)

    def _build_completion_kwargs(
        self,
        messages: list[dict],
        tool_schema: list[dict] | None,
        tool_choice: str,
        response_format: dict | object | None,
        suppress_thinking: bool,
        model: str,
    ) -> dict:
        completion_kwargs = {
            "model": model,
            "messages": messages,
            "tools": tool_schema,
            "tool_choice": tool_choice if tool_schema else None,
            "response_format": response_format,
        }
        if suppress_thinking and model.startswith(("ollama/", "ollama_chat/")):
            completion_kwargs.update({"think": False, "drop_params": True})
        provider = model.split("/", 1)[0].lower()
        primary_provider = self.llm_model.split("/", 1)[0].lower()
        if provider in ("ollama", "ollama_chat"):
            completion_kwargs["api_base"] = (
                self.api_base
                if primary_provider in ("ollama", "ollama_chat") and self.api_base
                else "http://localhost:11434"
            )
        elif model == self.llm_model and self.api_base:
            completion_kwargs["api_base"] = self.api_base
        if _ONE_SHOT_COMPLETION.get():
            completion_kwargs.update(
                {"num_retries": 0, "max_retries": 0, "fallbacks": []}
            )
        return completion_kwargs

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=60),
        retry=retry_if_exception(_should_retry_completion),
        reraise=True,
    )
    def generate(
        self,
        prompt: str | list[str] | None = None,
        tool_schema: list[dict] | None = None,
        tool_choice: str = "auto",
        response_format: dict | object | None = None,
        system_prompt: str | None = None,
        suppress_thinking: bool = False,
    ) -> str:
        """
        Generate a response from the LLM using litellm based on the prompt

        Args:
            prompt: The prompt to generate a response for (str, list of strings, or None)
            tool_schema: The schema of the tools to use
            tool_choice: The choice of tool to use
            response_format: The format of the response
            system_prompt: Optional system prompt scoped to this call only.
            suppress_thinking: Ask supported Ollama providers to return only
                final response content for this call.

        Returns:
            The response from the LLM
        """

        messages = self._build_messages(prompt, system_prompt=system_prompt)
        model_to_use = self.active_model
        completion_kwargs = self._build_completion_kwargs(
            messages,
            tool_schema,
            tool_choice,
            response_format,
            suppress_thinking,
            model_to_use,
        )

        try:
            response = completion(**completion_kwargs)
            response = self._validate_response(
                response,
                response_format,
                tool_schema,
            )
            logger.info("LLM request handled by model: %s", model_to_use)
            return response
        except Exception as error:
            if (
                self.fallback_model
                and not self._fallback_active
                and model_to_use == self.llm_model
            ):
                self._switch_to_fallback()
                try:
                    fallback_kwargs = self._build_completion_kwargs(
                        messages,
                        tool_schema,
                        tool_choice,
                        response_format,
                        suppress_thinking,
                        self.fallback_model,
                    )
                    response = completion(**fallback_kwargs)
                    response = self._validate_response(
                        response,
                        response_format,
                        tool_schema,
                    )
                    logger.info(
                        "LLM request handled by fallback model: %s",
                        self.fallback_model,
                    )
                    return response
                except Exception:
                    logger.exception(
                        "Fallback model %s failed",
                        self.fallback_model,
                    )
                    raise

            if str(error).startswith("This model isn't mapped yet."):
                raise self._build_invalid_model_error(error) from error
            if isinstance(error, RateLimitError):
                raise self._build_rate_limit_error(error) from error
            if isinstance(error, NotFoundError):
                raise self._build_invalid_model_error(error) from error
            raise

    async def agenerate(
        self,
        prompt: str | list[str] | None = None,
        tool_schema: list[dict] | None = None,
        tool_choice: str = "auto",
        response_format: dict | object | None = None,
        system_prompt: str | None = None,
        suppress_thinking: bool = False,
    ) -> str:
        """
        Asynchronous version of generate() method for parallel LLM calls.
        """
        messages = self._build_messages(prompt, system_prompt=system_prompt)
        async for attempt in AsyncRetrying(
            wait=wait_exponential(multiplier=1, min=1, max=60),
            retry=retry_if_exception(_should_retry_completion),
            reraise=True,
        ):
            with attempt:
                model_to_use = self.active_model
                completion_kwargs = self._build_completion_kwargs(
                    messages,
                    tool_schema,
                    tool_choice,
                    response_format,
                    suppress_thinking,
                    model_to_use,
                )

                try:
                    response = await acompletion(**completion_kwargs)
                    response = self._validate_response(
                        response,
                        response_format,
                        tool_schema,
                    )
                    logger.info("LLM request handled by model: %s", model_to_use)
                    return response
                except Exception as error:
                    if (
                        self.fallback_model
                        and not self._fallback_active
                        and model_to_use == self.llm_model
                    ):
                        self._switch_to_fallback()
                        try:
                            fallback_kwargs = self._build_completion_kwargs(
                                messages,
                                tool_schema,
                                tool_choice,
                                response_format,
                                suppress_thinking,
                                self.fallback_model,
                            )
                            response = await acompletion(**fallback_kwargs)
                            response = self._validate_response(
                                response,
                                response_format,
                                tool_schema,
                            )
                            logger.info(
                                "LLM request handled by fallback model: %s",
                                self.fallback_model,
                            )
                            return response
                        except Exception:
                            logger.exception(
                                "Fallback model %s failed",
                                self.fallback_model,
                            )
                            raise

                    # Treat unmapped-model message from litellm as an invalid model
                    if str(error).startswith("This model isn't mapped yet."):
                        raise self._build_invalid_model_error(error) from error

                    if isinstance(error, RateLimitError):
                        raise self._build_rate_limit_error(error) from error
                    if isinstance(error, NotFoundError):
                        raise self._build_invalid_model_error(error) from error
                    raise
        return response

    @staticmethod
    def _validate_response(response, response_format, tool_schema):
        if response_format is None or tool_schema:
            return response

        choices = (
            response.get("choices")
            if isinstance(response, dict)
            else getattr(response, "choices", None)
        )

        if not choices:
            raise ValueError("LLM returned no choices.")

        message = (
            choices[0].get("message")
            if isinstance(choices[0], dict)
            else getattr(choices[0], "message", None)
        )

        if message is None:
            raise ValueError("LLM returned no message.")

        content = (
            message.get("content")
            if isinstance(message, dict)
            else getattr(message, "content", None)
        )
        parsed = (
            message.get("parsed")
            if isinstance(message, dict)
            else getattr(message, "parsed", None)
        )

        if content is None and parsed is None:
            raise ValueError("LLM returned an empty structured response.")

        return response

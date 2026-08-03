import logging
import os

from dotenv import load_dotenv
from litellm import acompletion, completion, litellm
from litellm.exceptions import (
    APIConnectionError,
    NotFoundError,
    RateLimitError,
    Timeout,
)
from tenacity import AsyncRetrying, retry, retry_if_exception_type, wait_exponential

RETRYABLE_EXCEPTIONS = (
    APIConnectionError,
    Timeout,
    RateLimitError,
)

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
        self.system_prompt = system_prompt

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

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=60),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
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

        completion_kwargs = {
            "model": self.llm_model,
            "messages": messages,
            "tools": tool_schema,
            "tool_choice": tool_choice if tool_schema else None,
            "response_format": response_format,
        }
        if suppress_thinking and self.llm_model.startswith(("ollama/", "ollama_chat/")):
            completion_kwargs.update({"think": False, "drop_params": True})
        if self.api_base:
            completion_kwargs["api_base"] = self.api_base

        try:
            response = completion(**completion_kwargs)
        except RateLimitError as error:
            raise self._build_rate_limit_error(error) from error
        except NotFoundError as error:
            raise self._build_invalid_model_error(error) from error
        except Exception as error:
            if str(error).startswith("This model isn't mapped yet."):
                raise self._build_invalid_model_error(error) from error
            raise

        return response

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
            retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
            reraise=True,
        ):
            with attempt:
                completion_kwargs = {
                    "model": self.llm_model,
                    "messages": messages,
                    "tools": tool_schema,
                    "tool_choice": tool_choice if tool_schema else None,
                    "response_format": response_format,
                }
                if suppress_thinking and self.llm_model.startswith(
                    ("ollama/", "ollama_chat/")
                ):
                    completion_kwargs.update({"think": False, "drop_params": True})
                if self.api_base:
                    completion_kwargs["api_base"] = self.api_base

                try:
                    response = await acompletion(**completion_kwargs)
                except RateLimitError as error:
                    raise self._build_rate_limit_error(error) from error
                except NotFoundError as error:
                    raise self._build_invalid_model_error(error) from error
                except Exception as error:
                    if str(error).startswith("This model isn't mapped yet."):
                        raise self._build_invalid_model_error(error) from error
                    raise
        return response

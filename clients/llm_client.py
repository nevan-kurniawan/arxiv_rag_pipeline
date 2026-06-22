import openai
from openai import OpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    wait_exponential,
    stop_after_attempt,
)
from datetime import datetime, timezone
from schemas.llm_models import LLMResponse, TokenDetails


class LLMClient:
    def __init__(
        self,
        provider: str,
        api_key: str,
        base_url: str,
    ):
        self.provider = provider
        self.llm_client = OpenAI(api_key=api_key, base_url=base_url)

    @retry(
        retry=retry_if_exception_type(openai.RateLimitError),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        stop=stop_after_attempt(6),
        before_sleep=lambda retry_state: print(
            f"Rate limited. Retrying in {retry_state.next_action.sleep} seconds..."
        ),
    )
    def prompt_llm(
        self,
        prompt: str,
        llm_model: str,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        include_reasoning: bool | None = None,
    ) -> LLMResponse:
        kwargs = {
            "model": llm_model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
        if include_reasoning is not None:
            kwargs["include_reasoning"] = include_reasoning
        response = self.llm_client.chat.completions.create(
            **kwargs,
        )

        usage = response.usage

        token_details = TokenDetails(
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
        )

        return LLMResponse(
            response=response.choices[0].message.content,
            provider=self.provider,
            model=llm_model,
            token_details=token_details,
            generated_at=datetime.now(timezone.utc),
        )

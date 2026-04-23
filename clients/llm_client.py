import openai
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

class LLMClient:
    def __init__(self, api_key: str):
        self.llm_client = OpenAI(
            api_key = api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
    
    @retry(
        retry=retry_if_exception_type(openai.RateLimitError),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(5)
    )
    def prompt_llm(self, prompt:str, llm_model: str = 'gemini-3.1-flash-lite-preview'):
        response = self.llm_client.chat.completions.create(
            model=llm_model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        return response
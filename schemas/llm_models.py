from pydantic import BaseModel
from openai.types.chat import ChatCompletion

class LLMResponse(BaseModel):
    provider: str
    response: ChatCompletion
    model: str
    
from pydantic import BaseModel
from datetime import datetime

class TokenDetails(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class LLMResponse(BaseModel):
    response: str
    provider: str
    model: str
    token_details: TokenDetails
    generated_at: datetime
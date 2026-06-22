from pydantic import BaseModel, Field
from datetime import datetime

class ArxivDocument(BaseModel):
    title: str
    categories: list[str]
    authors: list[str]
    summary: str
    entry_id: str
    published: datetime

class SearchSyntheticGroundTruth(BaseModel):
    entry_id: str
    question: list[str]
    question_generated_at: datetime
    question_generated_by: str

class ResponseSyntheticGroundTruth(SearchSyntheticGroundTruth):
    retrieved_context: list[dict]
    response: str
    response_generated_by: str

class JudgementRecord(ResponseSyntheticGroundTruth):
    explanation_faithfulness: str
    faithfulness: int = Field(ge=1,le=3)
    explanation_relevance: str
    relevance: int = Field(ge=1,le=3)
    evaluation_generated_by: str
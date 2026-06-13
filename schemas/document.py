from pydantic import BaseModel
from datetime import datetime

class ArxivDocument(BaseModel):
    """_summary_

    Args:
        BaseModel (_type_): _description_
    """
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
from pydantic import BaseModel, Field
from typing import Literal, Optional

# Structured schema for Gemini classification
class QueryAnalysis(BaseModel):
    sentiment: Literal["Positive", "Neutral", "Negative"] = Field(
        ..., description="Sentiment classification of the customer query"
    )
    human_intervention_required: bool = Field(
        ..., description="True if query requires human intervention, False otherwise"
    )

class CustomerQueryRequest(BaseModel):
    query: str = Field(..., description="Customer query string.")
    provider: Optional[str] = Field("gemini", description="Options: 'gemini', 'base', 'finetuned'")

class CustomerQueryResponse(BaseModel):
    query: str
    sentiment: str
    human_intervention_required: bool
    answer: str
    model_used: str
    provider: str
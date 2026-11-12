from pydantic import BaseModel, Field
from typing import Literal, Optional, List

class QueryAnalysis(BaseModel):
    sentiment: Literal["Positive", "Neutral", "Negative"] = Field(
        ..., description="Sentiment classification of the customer query"
    )
    human_intervention_required: bool = Field(
        ..., description="True if query requires human intervention, False otherwise"
    )

# Schema for Router Decision
class RouterDecision(BaseModel):
    complexity: Literal["Simple", "Complex"] = Field(
        ..., description="Classify query complexity. 'Simple' for routine questions, 'Complex' for multi-part, policy edge cases, or ambiguous queries."
    )
    reasoning: str = Field(..., description="Brief reason for the complexity rating.")

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"] = Field(..., description="Role of the message author")
    content: str = Field(..., description="Message text content")

class CustomerQueryRequest(BaseModel):
    query: str = Field(..., description="Customer query string.")
    provider: Optional[str] = Field("gemini", description="Options: 'gemini', 'base', 'finetuned', 'router'")
    use_rag: bool = Field(False, description="Enable RAG context retrieval")
    mode: Optional[str] = Field("vanilla", description="Options: 'vanilla' or 'workplace'")
    chat_history: Optional[List[ChatMessage]] = Field(default=[], description="Prior multi-turn chat history")

class CustomerQueryResponse(BaseModel):
    query: str
    sentiment: str
    human_intervention_required: bool
    answer: str
    model_used: str
    provider: str
    rag_enabled: bool
    mode: str
    query_complexity: Optional[str] = Field(None, description="'Simple' or 'Complex' if router was used")
    routed_to: Optional[str] = Field(None, description="Actual execution target selected by the router")
from pydantic import BaseModel, Field
from typing import Optional

class CustomerQueryRequest(BaseModel):
    query: str = Field(..., description="Customer query string.")
    # Options: 'gemini', 'base', 'finetuned'
    provider: Optional[str] = Field("gemini", description="Model provider/variant choice.")

class CustomerQueryResponse(BaseModel):
    query: str
    answer: str
    model_used: str
    provider: str
from pydantic import BaseModel, Field

class CustomerQueryRequest(BaseModel):
    query: str = Field(..., description="The query string submitted by the customer.", example="How do I track my order?")

class CustomerQueryResponse(BaseModel):
    query: str
    answer: str
    model_used: str
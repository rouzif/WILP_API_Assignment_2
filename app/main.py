from fastapi import FastAPI, HTTPException, status
from app.schemas import CustomerQueryRequest, CustomerQueryResponse
from app.services.gemini import gemini_service
from app.config import settings

app = FastAPI(
    title="Customer Query Support API",
    description="API Endpoint powered by Gemini to answer customer queries.",
    version="1.0.0"
)


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Customer Support API is running"}


@app.post(
    "/api/v1/customer-query",
    response_model=CustomerQueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Answer Customer Query"
)
async def process_customer_query(request: CustomerQueryRequest):
    if not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty."
        )

    try:
        answer = gemini_service.generate_answer(query=request.query)
        return CustomerQueryResponse(
            query=request.query,
            answer=answer,
            model_used=settings.DEFAULT_MODEL
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating response from Gemini API: {str(e)}"
        )
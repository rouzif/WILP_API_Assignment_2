from fastapi import FastAPI, HTTPException, status
from app.schemas import CustomerQueryRequest, CustomerQueryResponse
from app.services.gemini import gemini_service
from app.services.slm import slm_service
from app.config import settings

app = FastAPI(title="Multi-Subtask Customer AI API")

@app.post("/api/v1/customer-query", response_model=CustomerQueryResponse)
async def process_customer_query(request: CustomerQueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    provider = (request.provider or "gemini").lower()
    use_rag = request.use_rag

    try:
        if provider == "base":
            result = slm_service.process_query_workflow(query=request.query, model_type="base", use_rag=use_rag)
            model_used = "Gemma 2B (Base)"
        elif provider == "finetuned":
            result = slm_service.process_query_workflow(query=request.query, model_type="finetuned", use_rag=use_rag)
            model_used = "Gemma 2B (Fine-Tuned)"
        else:
            result = gemini_service.process_query_workflow(query=request.query, use_rag=use_rag)
            model_used = settings.DEFAULT_MODEL
            provider = "gemini"

        return CustomerQueryResponse(
            query=request.query,
            sentiment=result["sentiment"],
            human_intervention_required=result["human_intervention_required"],
            answer=result["answer"],
            model_used=model_used,
            provider=provider,
            rag_enabled=use_rag
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing workflow on provider '{provider}': {str(e)}"
        )
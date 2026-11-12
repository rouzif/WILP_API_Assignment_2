from fastapi import FastAPI, HTTPException
from app.schemas import CustomerQueryRequest, CustomerQueryResponse
from app.services.gemini import gemini_service
from app.services.slm import slm_service
from app.config import settings

app = FastAPI(title="Multi-Subtask AI Support API")

@app.post("/api/v1/customer-query", response_model=CustomerQueryResponse)
def process_customer_query(request: CustomerQueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    provider = (request.provider or "gemini").lower()
    use_rag = request.use_rag
    mode = request.mode or "vanilla"
    history = [msg.model_dump() for msg in request.chat_history] if request.chat_history else []

    query_complexity = None
    routed_to = None

    try:
        # 1. Always execute Subtasks 1 (Sentiment) & 2 (Escalation) via Gemini
        analysis = gemini_service.analyze_query(query=request.query)
        sentiment = analysis["sentiment"]
        needs_human = analysis["human_intervention_required"]

        # 2. Execute Subtask 3 based on selected provider
        if provider == "router":
            query_complexity = gemini_service.classify_complexity(query=request.query)

            if query_complexity == "Simple":
                routed_to = "Fine-Tuned Gemma 2B (SLM)"
                result = slm_service.process_query_workflow(
                    query=request.query,
                    model_type="finetuned",
                    use_rag=use_rag,
                    domain=mode,
                    chat_history=history,
                    sentiment=sentiment,
                    needs_human=needs_human
                )
                model_used = "Dynamic Router -> Gemma 2B (Fine-Tuned)"
            else:
                routed_to = f"{settings.DEFAULT_MODEL} (Cloud LLM)"
                result = gemini_service.process_query_workflow(
                    query=request.query,
                    use_rag=use_rag,
                    domain=mode,
                    chat_history=history,
                    sentiment=sentiment,
                    needs_human=needs_human
                )
                model_used = f"Dynamic Router -> {settings.DEFAULT_MODEL}"

        elif provider == "base":
            result = slm_service.process_query_workflow(
                query=request.query,
                model_type="base",
                use_rag=use_rag,
                domain=mode,
                chat_history=history,
                sentiment=sentiment,
                needs_human=needs_human
            )
            model_used = "Gemma 2B (Base)"

        elif provider == "finetuned":
            result = slm_service.process_query_workflow(
                query=request.query,
                model_type="finetuned",
                use_rag=use_rag,
                domain=mode,
                chat_history=history,
                sentiment=sentiment,
                needs_human=needs_human
            )
            model_used = "Gemma 2B (Fine-Tuned)"

        else:
            provider = "gemini"
            result = gemini_service.process_query_workflow(
                query=request.query,
                use_rag=use_rag,
                domain=mode,
                chat_history=history,
                sentiment=sentiment,
                needs_human=needs_human
            )
            model_used = settings.DEFAULT_MODEL

        return CustomerQueryResponse(
            query=request.query,
            sentiment=result["sentiment"],
            human_intervention_required=result["human_intervention_required"],
            answer=result["answer"],
            model_used=model_used,
            provider=provider,
            rag_enabled=use_rag,
            mode=mode,
            query_complexity=query_complexity,
            routed_to=routed_to
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error executing provider '{provider}': {str(e)}"
        )
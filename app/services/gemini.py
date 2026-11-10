import json
from google import genai
from google.genai import types
from app.config import settings
from app.schemas import QueryAnalysis, RouterDecision
from app.services.rag import rag_service


class GeminiService:
    def __init__(self):
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)

    def classify_complexity(self, query: str, model: str = settings.DEFAULT_MODEL) -> str:
        prompt = (
            f"Analyze this customer support query:\n\"{query}\"\n\n"
            "Classify it as either 'Simple' or 'Complex'.\n"
            "- 'Simple': Basic standard queries (e.g., store hours, standard return timeframe, basic shipping fee, simple greetings).\n"
            "- 'Complex': Multi-part questions, edge cases, complaints requiring nuanced judgment, or policy exceptions."
        )

        try:
            response = self.client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=RouterDecision,
                    temperature=0.0
                )
            )
            decision = RouterDecision.model_validate_json(response.text)
            return decision.complexity
        except Exception:
            return "Complex"  # Fallback to Complex on failure

    def process_query_workflow(
            self,
            query: str,
            model: str = settings.DEFAULT_MODEL,
            use_rag: bool = False,
            domain: str = "vanilla"
    ):
        context = ""
        if use_rag:
            context = rag_service.retrieve_context(query, domain=domain)

        # Sentiment & Escalation Analysis
        analysis_prompt = (
            f"Analyze query: \"{query}\". Classify sentiment (Positive, Neutral, Negative) "
            "and decide if human intervention is required (true/false)."
        )

        analysis_response = self.client.models.generate_content(
            model=model,
            contents=analysis_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=QueryAnalysis,
                temperature=0.1
            )
        )

        analysis = QueryAnalysis.model_validate_json(analysis_response.text)
        sentiment = analysis.sentiment
        needs_human = analysis.human_intervention_required

        # Subtask 3 Response Generation
        if sentiment == "Positive" and needs_human:
            answer = "Thank you! We are routing your query to a human chat agent."
        elif sentiment == "Negative" and needs_human:
            answer = "We sincerely apologize for your experience. We are routing your issue to an account manager."
        else:
            if use_rag and context:
                prompt = (
                    "You are a helpful customer support agent.\n"
                    "STRICT INSTRUCTIONS: Answer the customer's question strictly using ONLY the provided policy context below. "
                    "Do NOT invent facts or placeholders. If the answer is not contained in the context, say 'I do not have information on that policy.'\n\n"
                    f"Official Policy Context:\n{context}\n\n"
                    f"Customer Question: {query}"
                )
            else:
                prompt = f"Customer Query: {query}\nProvide a helpful customer support response."

            res = self.client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    http_options=types.HttpOptions(timeout=30000)
                )
            )
            answer = res.text.strip()

        return {
            "sentiment": sentiment,
            "human_intervention_required": needs_human,
            "answer": answer
        }


gemini_service = GeminiService()
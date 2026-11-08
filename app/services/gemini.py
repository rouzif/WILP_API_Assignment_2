import json
from google import genai
from google.genai import types
from app.config import settings
from app.schemas import QueryAnalysis


class GeminiService:
    def __init__(self):
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)

    def process_query_workflow(self, query: str, model: str = settings.DEFAULT_MODEL):
        # 1. Perform Subtask 1 & 2 using Structured Outputs
        analysis_prompt = (
            f"Analyze the following customer query:\n\n\"{query}\"\n\n"
            "1. Classify sentiment as Positive, Neutral, or Negative.\n"
            "2. Determine if human intervention is required (True/False)."
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

        # 3. Subtask 3 Logic & Response Generation
        if sentiment == "Positive" and needs_human:
            answer = "Thank you for reaching out! We appreciate your positive feedback. We are routing your query to a human chat agent right away."

        elif sentiment == "Positive" and not needs_human:
            gen_prompt = f"The customer query is positive: '{query}'. Thank the customer and provide a helpful answer."
            res = self.client.models.generate_content(model=model, contents=gen_prompt)
            answer = res.text.strip()

        elif sentiment == "Negative" and needs_human:
            answer = "We sincerely apologize for the experience you have faced. Your issue is being prioritized, and we are routing your request directly to an account manager."

        elif sentiment == "Negative" and not needs_human:
            gen_prompt = f"The customer query is negative: '{query}'. Apologize to the customer and respond to address their concern."
            res = self.client.models.generate_content(model=model, contents=gen_prompt)
            answer = res.text.strip()

        else:  # Neutral sentiment cases
            if needs_human:
                answer = "Thank you for your query. We are routing your message to a customer support agent for assistance."
            else:
                gen_prompt = f"Respond to this customer query concisely and helpfully: '{query}'"
                res = self.client.models.generate_content(model=model, contents=gen_prompt)
                answer = res.text.strip()

        return {
            "sentiment": sentiment,
            "human_intervention_required": needs_human,
            "answer": answer
        }


gemini_service = GeminiService()
from google import genai
from app.config import settings


class GeminiService:
    def __init__(self):
        # Automatically picks up GEMINI_API_KEY from environment or settings
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)

    def generate_answer(self, query: str, model: str = settings.DEFAULT_MODEL) -> str:
        system_instruction = (
            "You are a helpful and polite customer support assistant. "
            "Provide accurate, clear, and concise answers to customer inquiries."
        )

        response = self.client.models.generate_content(
            model=model,
            contents=query,
            config={
                "system_instruction": system_instruction,
                "temperature": 0.3,
            }
        )
        return response.text


gemini_service = GeminiService()
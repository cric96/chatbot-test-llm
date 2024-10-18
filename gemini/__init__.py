from core import LanguageModelProvider, LanguageModel
import requests
import json


import os
import google.generativeai as genai

class GeminiService(LanguageModelProvider):
    def __init__(self, api_key_env: str="GENAI_API_KEY"):
        api_key = os.getenv(api_key_env)
        genai.configure(api_key=api_key)

    def use(self, language_model: str, system_prompt: str) -> LanguageModel:
        return GeminiLanguageModel(language_model, system_prompt)


class GeminiLanguageModel(LanguageModel):
    def __init__(self, name: str, system: str):
        self.name = name
        self.system = system

        self.generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }
        self.model = genai.GenerativeModel(
            model_name=self.name,
            generation_config=self.generation_config,
            system_instruction=self.system
        )

    def ask(self, question: str) -> str:

        chat_session = self.model.start_chat(
            history=[]  # Use context for system prompt
        )
        response = chat_session.send_message(question)
        return response.text
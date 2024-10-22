from deepeval.models.base_model import DeepEvalBaseLLM
import os
import google.generativeai as genai

class GoogleVertexAI(DeepEvalBaseLLM):
    """Class to implement Vertex AI for DeepEval"""
    def __init__(self, model, env="GEMINI_API_KEY", *args, **kwargs):
#        super().__init__(*args, **kwargs)
        genai.configure(api_key=os.environ[env])
        self.model = model
        self.generation_config = {
          "temperature": 1,
          "top_p": 0.95,
          "top_k": 40,
          "max_output_tokens": 8192,
          "response_mime_type": "text/plain",
        }


    def load_model(self):
        return genai.GenerativeModel(
            model_name=self.model,
            generation_config=self.generation_config,
        )

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        chat = chat_model.start_chat(history=[])
        return chat.send_message(prompt).text

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        chat = chat_model.start_chat(history=[])
        return chat.send_message(prompt).text

    def get_model_name(self):
        return "Vertex AI Model " + self.model

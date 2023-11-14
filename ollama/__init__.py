from core import LanguageModelProvider, LanguageModel
import requests
import json


# A language model provider that uses the Ollama API to generate text.
class OllamaService(LanguageModelProvider):
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port

    def use(self, language_model: str, system_prompt: str) -> LanguageModel:
        return OllamaLanguageModel(self.host, self.port, language_model, system_prompt)


# A language model that uses the Ollama API to generate text based on the REST API described here: https://github.com/jmorganca/ollama/blob/main/docs/api.md
class OllamaLanguageModel(LanguageModel):
    def __init__(self, host: str, port: int, name: str, system_prompt: str):
        self.host = host
        self.port = port
        self.system_prompt = system_prompt
        self.name = name

    def ask(self, question: str) -> str:
        payload = {
            'model': self.name,
            'prompt': f'{question}',
            'stream': False,
            'system_prompt': self.system_prompt,
        }
        reply = requests.post(f'http://{self.host}:{self.port}/api/generate', data=json.dumps(payload))
        return reply.json()['response']

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


# A language model that uses the Ollama API to generate text based on the REST API
# described here: https://github.com/jmorganca/ollama/blob/main/docs/api.md
class OllamaLanguageModel(LanguageModel):
    def __init__(self, host: str, port: int, name: str, system: str):
        self.host = host
        self.port = port
        self.system = system
        self.name = name

    def ask(self, question: str) -> str:
        payload = {
            'model': self.name,
            'prompt': f'{question}',
            'stream': False,
            'system': self.system,
        }
        reply = requests.post(f'http://{self.host}:{self.port}/api/generate', data=json.dumps(payload))
        return OllamaLanguageModel._pretty_format(reply.json()['response'])

    @staticmethod
    def _pretty_format(response: str) -> str:
        # remove all double quotes
        response = response.replace('"', '')
        # add a double quote at the beginning and end
        response = f'"{response}"'
        return response

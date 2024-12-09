from abc import ABC, abstractmethod


# Abstract class for language models
class LanguageModel(ABC):
    @abstractmethod
    def ask(self, question: str, max_output: int) -> str:
        pass


class LanguageModelProvider(ABC):
    @abstractmethod
    def use(self, language_model: str, system: str) -> LanguageModel:
        pass

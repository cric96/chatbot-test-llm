from core import LanguageModelProvider
from collections.abc import Iterable
from pydoc import locate


class BenchTarget:
    def __init__(self, provider: LanguageModelProvider, models: list[str], system_prompt: str):
        self.provider = provider
        self.models = models
        self.system_prompt = system_prompt


class Result:
    def __init__(self, output: str, expected: str):
        self.output = output
        self.expected = expected
        self.correct = output == expected


def verify(target: BenchTarget, knowledge: Iterable[(str, str)]) -> Iterable[(str, list[Result])]:
    # for each knowledge pair, ask each model
    result = []
    models = [target.provider.use(model, target.system_prompt) for model in target.models]
    for (question, expected) in knowledge:
        results = []
        for model in models:
            output = model.ask(question)
            results.append(Result(output, expected))
        result.append((question, results))
    return result


def target_from_object(obj: dict) -> BenchTarget:
    provider_class = locate(obj['provider']['name'])
    provider = provider_class(**obj['provider']['args'])
    models = obj['models']
    system_prompt = obj['system_prompt']
    return BenchTarget(provider, models, system_prompt)

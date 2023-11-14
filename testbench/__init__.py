import csv
from hashlib import md5
from pathlib import Path
from core import LanguageModelProvider, LanguageModel
from collections.abc import Iterable
from pydoc import locate
from ._logging import *


PATH = Path(__file__).parents[0]
CACHE = PATH / 'cache'


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

    def __str__(self):
        return f'Output: {self.output}, Expected: {self.expected}, Correct: {self.correct}'

    def to_csv(self):
        return f'"{self.output}", "{self.expected}"'


def evaluate_target(target: BenchTarget, knowledge: Iterable[(str, str)], use_cache: bool = True) -> Iterable[(str, list[Result])]:

    def ask_model(local_model: LanguageModel, local_question: str) -> str:
        logger.debug(f'Asking "{local_question}"')
        return local_model.ask(local_question)

    # for each knowledge pair, ask each model
    result = []
    models = [target.provider.use(model, target.system_prompt) for model in target.models]
    for (question, expected) in knowledge:
        responses = []
        if use_cache:
            if CACHE.exists() is False:
                CACHE.mkdir()
            hyperparameters = f'{question}_{target.models}_{target.system_prompt}'
            hash_file_name = md5(hyperparameters.encode()).hexdigest() + '.csv'
            file_name = CACHE / hash_file_name
            if file_name.is_file():
                # read from cache
                with open(file_name, 'r') as f:
                    # for each line create a result
                    reader = csv.reader(f)
                    for line in reader:
                        output, expected = line
                        responses.append(Result(output, expected))
                result.append((question, responses))
                continue
            # ask model
            else:
                enable_file_logging(str(file_name))
                for model in models:
                    reply = Result(ask_model(model, question), expected)
                    responses.append(result)
                    logger.info(f'{reply.to_csv()}')
            disable_file_logging()
        else:
            for model in models:
                responses.append(Result(ask_model(model, question), expected))
        result.append((question, responses))
    return result


def target_from_object(obj: dict) -> BenchTarget:
    provider_class = locate(obj['provider']['name'])
    provider = provider_class(**obj['provider']['args'])
    models = obj['models']
    system_prompt = obj['system_prompt']
    return BenchTarget(provider, models, system_prompt)

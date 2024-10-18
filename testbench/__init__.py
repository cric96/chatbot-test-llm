from __future__ import annotations
import csv
import random
import re
from hashlib import md5
from pathlib import Path
from core import LanguageModelProvider, LanguageModel
from collections.abc import Iterable
from pydoc import locate
from .logging import *
import io
from evaluate import load


PATH = Path(__file__).parents[0]
CACHE = PATH / 'cache'
BERT_SCORE = load('bertscore')
METEOR = load('meteor')

def csv_formatter(string):
    outstream = io.StringIO()  # "fake" output file
    cw = csv.writer(outstream, quoting=csv.QUOTE_ALL, lineterminator="")  # pass the fake file to csv module
    cw.writerow([string])  # write a row
    return outstream.getvalue()


class BenchTarget:
    def __init__(self, provider: LanguageModelProvider, models: list[str], system: str, classes: list[str] = None):
        self.provider = provider
        self.models = models
        self.system = system
        self.classes = classes

    def new_target(self, model: str | list[str]) -> BenchTarget:
        if isinstance(model, str):
            return BenchTarget(self.provider, [model], self.system, self.classes)
        elif isinstance(model, list):
            return BenchTarget(self.provider, model, self.system, self.classes)


class Result:
    def __init__(self, output: str, expected: str):
        self._output = output.strip()
        self._expected = expected.strip()

    @property
    def output(self) -> str:
        return self._output

    @property
    def clean_output(self) -> str:
        return self._output

    @property
    def expected(self) -> str:
        return self._expected

    def clean_expected(self) -> str:
        return self._expected

    @property
    def correct(self) -> bool:
        return self.clean_output == self.clean_expected

    @property
    def one_by_one_comparison(self) -> tuple[bool, bool, bool]:
        raise NotImplementedError

    def bert_comparison(self):
        return BERT_SCORE.compute(predictions=[self.output], references=[self.expected], lang="it")
    def meteor_comparison(self):
        return METEOR.compute(predictions=[self.output], references=[self.expected])
    def __repr__(self):
        return f'Output: {self.output}, Expected: {self.expected}, Correct: {self.correct}'

    def to_csv(self):
        return f'{csv_formatter(self.output)} {csv_formatter(self.expected)}'


class SmartResult(Result):

    @staticmethod
    def _clean_string(string: str) -> str:
        return re.sub("[^A-Za-z0-9-]+", "", string).lower()

    @staticmethod
    def clean_comparison(first: str, second: str) -> bool:
        return SmartResult._clean_string(first) == SmartResult._clean_string(second)

    @property
    def output(self) -> str:
        return self._output

    @property
    def clean_output(self) -> str:
        return self._clean_string(self._output)

    @property
    def expected(self) -> str:
        return self._expected

    @property
    def clean_expected(self) -> str:
        return self._clean_string(self._expected)


class RequestResult(SmartResult):
    legal_measures = ['frequenza', 'pressione', 'entrambi', 'generale']
    legal_formats = ['media', 'lista', 'grafico']
    default_measure = 'generale'
    default_quantity = '-1'
    default_format = 'lista'

    def _get_output(self, key: int) -> str:
        ## if contains the key, return it or none
        if len(self._output.split(" ")) > key:
            return self._output.split(" ")[key]
        else:
            return ""

    def _get_expected(self, key: int) -> str:
        return self._expected.split(" ")[key]

    @property
    def measure_comparison(self) -> bool:
        output_measure = self._clean_string(self._get_output(0))
        if output_measure not in self.legal_measures:
            output_measure = self.default_measure
        return self.clean_comparison(output_measure, self._get_expected(0))

    @property
    def quantity_comparison(self) -> bool:
        output_quantity = self._clean_string(self._get_output(1))
        # check if output quantity is a number
        try:
            int(output_quantity)
        except ValueError:
            output_quantity = self.default_quantity
        return self.clean_comparison(output_quantity, self._get_expected(1))

    @property
    def format_comparison(self) -> bool:
        output_format = self._clean_string(self._get_output(2))
        if output_format not in self.legal_formats:
            output_format = self.default_format
        return self.clean_comparison(output_format, self._get_expected(2))

    @property
    def one_by_one_comparison(self) -> tuple[bool, bool, bool]:
        return self.measure_comparison, self.quantity_comparison, self.format_comparison

    @property
    def correct(self) -> bool:
        return self.measure_comparison and self.quantity_comparison and self.format_comparison


def evaluate_target(target: BenchTarget,
                    knowledge: Iterable[(str, str)],
                    use_cache: bool = True,
                    classification: bool = False) -> Iterable[(str, list[Result])]:
    def ask_model(local_model: LanguageModel, local_question: str) -> str:
        logger.debug(f'Asking "{local_question}"')
        return local_model.ask(local_question)

    # for each knowledge pair, ask each model
    result_class = SmartResult if classification else RequestResult
    result = []
    models = [target.provider.use(model, target.system) for model in target.models]
    classes = target.classes
    for element in knowledge:
        question = element[0]
        expected = element[1]
        responses = []
        if use_cache:
            if CACHE.exists() is False:
                CACHE.mkdir()
            hyperparameters = f'{question}_{target.models}_{target.system}'
            hash_file_name = md5(hyperparameters.encode()).hexdigest() + '.csv'
            file_name = CACHE / hash_file_name
            if file_name.is_file():
                # read from ca
                with open(file_name, 'r') as f:
                    # for each line create a result
                    reader = csv.reader(f, delimiter=' ')
                    for idx, line in enumerate(reader):
                        if len(line) != 2:
                            message = f'Invalid line in cache file: {line}\n in file {file_name}, line {idx}, length {len(line)}'
                            raise ValueError(message)
                        output, expected = line
                        responses.append(result_class(output, expected))
                result.append((question, responses))
                if len(responses) != len(models):
                    raise ValueError(f'Cache file {file_name} has {len(responses)} responses, but {len(models)} are expected')
                continue
            # ask model
            else:
                enable_file_logging(str(file_name))
                for model in models:
                    reply = Result(ask_model(model, question), expected)
                    # Check if reply is a valid class
                    if classes is not None:
                        if result_class._clean_string(reply.output) not in classes:
                            # Retry
                            reply = Result(ask_model(model, question), expected)
                            # Check if reply is a valid class
                            if result_class._clean_string(reply.output) not in classes:
                                # Randomly pick a class
                                reply = result_class(random.choice(classes), expected)
                    responses.append(reply)
                    logger.info(f'{reply.to_csv()}')
                result.append((question, responses))
            disable_file_logging()
        else:
            for model in models:
                responses.append(result_class(ask_model(model, question), expected))
            result.append((question, responses))
    return result


def target_from_object(obj: dict) -> BenchTarget:
    provider_class = locate(obj['provider']['name'])
    provider = provider_class(**obj['provider']['args'])
    models = obj['models']
    system = obj['system']
    classes = obj['classes']
    return BenchTarget(provider, models, system, classes)

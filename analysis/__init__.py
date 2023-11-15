import csv
import re
from hashlib import md5
from typing import Iterable, Type
from testbench import Result, BenchTarget, CACHE


class SmartResult(Result):

    @property
    def correct(self) -> bool:
        # remove all non-alphanumeric characters (e.g., spaces, punctuation)
        return re.sub("[^A-Za-z0-9]+", "", self.output) == re.sub("[^A-Za-z0-9]+", "", self.expected)


class Statistics:
    def __init__(self, results: Iterable[Type[Result]]):
        self.results = results

    @property
    def accuracy(self) -> float:
        return len([result for result in self.results if result.correct]) / len(self.results)


def analise_target(target: BenchTarget, knowledge: Iterable[tuple[str, str]]) -> Iterable[Statistics]:
    # for each knowledge pair, ask each model
    model_names = [model for model in target.models]
    results = {model: [] for model in model_names}
    for (question, expected) in knowledge:
        hyperparameters = f'{question}_{target.models}_{target.system}'
        hash_file_name = md5(hyperparameters.encode()).hexdigest() + '.csv'
        file_name = CACHE / hash_file_name
        with open(file_name, 'r') as f:
            # for each line create a result
            reader = csv.reader(f)
            for idx, line in enumerate(reader):
                if len(line) == 1:
                    # this means that in the model's response there were nested quotes
                    # So we need to merge the lines
                    next_line = next(reader)
                    line = line[0] + next_line[0], next_line[1]
                if len(line) != 2:
                    raise ValueError(f'Invalid line in cache file: {line}')
                output, expected = line
                results[model_names[idx]].append(SmartResult(output, expected))
    return [Statistics(result_list) for result_list in results.values()]
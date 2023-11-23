import csv
import pandas as pd
from hashlib import md5
from typing import Iterable, Type
from testbench import Result, BenchTarget, CACHE, SmartResult


class Statistics:
    def __init__(self, results: Iterable[Type[Result]]):
        self.results = results

    @property
    def accuracy(self) -> float:
        return len([result for result in self.results if result.correct]) / len(self.results)


    @property
    def confusion_matrix(self) -> pd.DataFrame:
        unique_categories = sorted(list(set([result.expected for result in self.results]))) + ['altro']
        matrix = pd.DataFrame(0, index=unique_categories, columns=unique_categories)
        for result in self.results:
            r_out = result.output
            if result.output not in unique_categories:
                r_out = 'altro'
            matrix.loc[result.expected, r_out] += 1
        return matrix


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
            content = f.read()
            content = content.replace('""', '"')
            content = content.replace("\\'", '')

            # for each line create a result
            reader = csv.reader(content.splitlines())

            for idx, line in enumerate(reader):
                if len(line) == 1:
                    # this means that in the model's response there were nested quotes
                    # So we need to merge the lines
                    next_line = next(reader)
                    if len(next_line) != 2:
                        raise ValueError(f'Invalid line in cache file: {line} in file {file_name}')
                    line = line[0] + next_line[0], next_line[1]
                if len(line) != 2:
                    raise ValueError(f'Invalid line in cache file: {line}')
                output, expected = line
                results[model_names[idx]].append(SmartResult(output, expected))
    return [Statistics(result_list) for result_list in results.values()]
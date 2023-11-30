import csv
import pandas as pd
from hashlib import md5
from typing import Iterable, Type
from testbench import Result, BenchTarget, CACHE, SmartResult, RequestResult


class Statistics:
    def __init__(self, results: Iterable[Type[Result]]):
        self.results = results

    @property
    def accuracy(self) -> float:
        return len([result for result in self.results if result.correct]) / len(self.results)

    @property
    def one_by_one_accuracy(self) -> tuple[float, float, float]:
        number_of_results = len(self.results)
        m, q, f = list(zip(*[result.one_by_one_comparison for result in self.results]))
        return sum(m) / number_of_results, sum(q) / number_of_results, sum(f) / number_of_results

    @property
    def confusion_matrix(self) -> pd.DataFrame:
        unique_categories = sorted(list(set([result.expected for result in self.results])))
        matrix = pd.DataFrame(0, index=unique_categories, columns=unique_categories)
        for result in self.results:
            r_out = result.output
            matrix.loc[result.expected, r_out] += 1
        return matrix


def analise_target(target: BenchTarget, knowledge: Iterable[tuple[str, str]], classification: bool = False) -> Iterable[Statistics]:
    result_class = SmartResult if classification else RequestResult
    # for each knowledge pair, ask each model
    model_names = [model for model in target.models]
    results = {model: [] for model in model_names}
    for element in knowledge:
        question = element[0]
        expected = element[1]
        hyperparameters = f'{question}_{target.models}_{target.system}'
        hash_file_name = md5(hyperparameters.encode()).hexdigest() + '.csv'
        file_name = CACHE / hash_file_name
        with open(file_name, 'r') as f:
            # for each line create a result
            reader = csv.reader(f, delimiter=' ')
            for idx, line in enumerate(reader):
                output, _ = line
                results[model_names[idx]].append(result_class(output, expected))
    return [Statistics(result_list) for result_list in results.values()]


def analise_request(knowledge: Iterable[tuple[str, str]]) -> Iterable[Statistics]:
    # for each knowledge pair, ask each model
    results = {'gpt': []}
    for (question, expected, gpt) in knowledge:
        # expected_measure, expected_quantity, expected_format = expected.split()
        if len(gpt.split()) != 3:
            results['gpt'].append(RequestResult('altro altro altro', expected))
            continue
        # pgt_measure, pgt_quantity, pgt_format = gpt.split()
        results['gpt'].append(RequestResult(gpt, expected))
    return [Statistics(result_list) for result_list in results.values()]

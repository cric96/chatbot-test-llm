import argparse

import pandas as pd
import yaml
import csv
import os
from testbench import target_from_object, evaluate_target


parser = argparse.ArgumentParser(description='LLM comparison for sentiment analysis in healthcare')
parser.add_argument('--data-file', type=str, default='./data/general/test.csv', help='input file path')
parser.add_argument('--bench-file', type=str, default='./data/general/bench.yml', help='benchmark configuration path')

if __name__ == '__main__':
    args = parser.parse_args()
    with open(os.path.abspath(args.data_file), 'r') as data_file:
        with open(os.path.abspath(args.bench_file), 'r') as bench_definition_file:
            bench = target_from_object(yaml.load(bench_definition_file, Loader=yaml.FullLoader)[0])
            reader = csv.reader(data_file)
            headers = next(reader)
            data = list(reader)
            targets = [bench.new_target(target) for target in bench.models]
            reports = [evaluate_target(target, data) for target in targets]
            questions, all_responses = [], []
            for i, report in enumerate(reports):
                model_responses = []
                for question, responses in report:
                    if i == 0:
                        questions.append(question)
                    for response in responses:
                        model_responses.append(response.output)
                all_responses.append(model_responses)
            general_responses = pd.DataFrame(all_responses).T
            general_responses = general_responses.assign(Question=questions)
            general_responses.columns = [target.models[0] for target in targets] + ['Question']
            # Move last columns as first
            cols = general_responses.columns.to_list()
            cols = cols[-1:] + cols[:-1]
            general_responses = general_responses[cols]
            general_responses.to_csv('general_responses.csv', index=False)

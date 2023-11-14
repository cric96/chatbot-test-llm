import argparse
import yaml
import csv
import os
from testbench import target_from_object, evaluate_target

parser = argparse.ArgumentParser(description='LLM comparison for sentiment analysis in healthcare')
parser.add_argument('--data-file', type=str, default='./data/sentences-short.csv', help='input file path')
parser.add_argument('--bench-file', type=str, default='./data/bench.yml', help='benchmark configuration path')

if __name__ == '__main__':
    args = parser.parse_args()
    with open(os.path.abspath(args.bench_file), 'r') as bench_definition_file:
        with open(os.path.abspath(args.data_file), 'r') as data_file:
            bench_list = yaml.load(bench_definition_file, Loader=yaml.FullLoader)
            reader = csv.reader(data_file)
            data = list(reader)
            targets = [target_from_object(bench) for bench in bench_list]
            reports = [evaluate_target(target, data) for target in targets]
            print(reports)


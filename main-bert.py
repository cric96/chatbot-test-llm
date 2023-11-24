import argparse
import os
import pandas as pd
import yaml
from evaluate import load
import numpy as np
from testbench import enable_logging, LOG_DEBUG, logger
parser = argparse.ArgumentParser(description='LLM comparison using BERTfor semantic similarity')
parser.add_argument('--reference-file', type=str, default='./data/general/general-test.csv', help='input file path')
parser.add_argument('--bench-file', type=str, default='./data/general/bench.yml', help='benchmark configuration path')
enable_logging(level=LOG_DEBUG)

if __name__ == '__main__':
    args = parser.parse_args()
    bert_score = load('bertscore')
    with open(os.path.abspath(args.reference_file), 'r') as reference_file:
        with open(os.path.abspath(args.bench_file), 'r') as bench_file:
            bench_list = yaml.load(bench_file, Loader=yaml.FullLoader)
            reference = pd.read_csv(reference_file)
            ground_truth = reference['Response']
            for bench in bench_list:
                name = bench['name']
                file = bench['file']
                test = pd.read_csv(file)
                results = bert_score.compute(predictions=ground_truth, references=test['Response'], lang="it")
                results.pop('hashcode')
                for metric, values in results.items():
                    mean_value = np.mean(np.array(values))
                    results[metric] = mean_value
                logger.info(f'{name} results: {results}')
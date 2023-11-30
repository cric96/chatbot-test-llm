import argparse
import yaml
import csv
import os
from analysis import analise_target
from testbench import target_from_object, evaluate_target, logger, enable_logging, LOG_INFO
from testbench.logging import INDENT, LOG_FLOAT_PRECISION


parser = argparse.ArgumentParser(description='LLM comparison for sentiment analysis in healthcare')
parser.add_argument('--data-file', type=str, default='./data/request/test.csv', help='input file path')
parser.add_argument('--bench-file', type=str, default='./data/request/bench.yml', help='benchmark configuration path')
enable_logging(level=LOG_INFO)

if __name__ == '__main__':
    args = parser.parse_args()
    with open(os.path.abspath(args.data_file), 'r') as data_file:
        with open(os.path.abspath(args.bench_file), 'r') as bench_definition_file:
            logger.info(f'Loading data...')
            bench = target_from_object(yaml.load(bench_definition_file, Loader=yaml.FullLoader)[0])
            reader = csv.reader(data_file)
            headers = next(reader)
            data = list(reader)
            targets = [bench.new_target(model) for model in bench.models]
            logger.debug(f'Loaded {len(targets)} target' + ('s' if len(targets) > 1 else ''))
            reports = [evaluate_target(target, data) for target in targets]
            for report in reports:
                for question, responses in report:
                    logger.debug(f'Question: {question}')
                    for response in responses:
                        logger.debug(f'{INDENT}Output: {response._output}')
                        logger.debug(f'{INDENT}Expected: {response._expected}')
            statistics = [analise_target(target, data) for target in targets]
            for idx, models_statistics in enumerate(statistics):
                for model_statistics in models_statistics:
                    measure, quantity, format = model_statistics.one_by_one_accuracy
                    logger.info(f'Results: for {targets[idx].models[0]}:'
                                f'\n\taccuracy: {model_statistics.accuracy :.{LOG_FLOAT_PRECISION}f}'
                                f'\n\tmeasure: {measure:.{LOG_FLOAT_PRECISION}f}'
                                f'\n\tquantity: {quantity:.{LOG_FLOAT_PRECISION}f}'
                                f'\n\tformat: {format:.{LOG_FLOAT_PRECISION}f}\n')

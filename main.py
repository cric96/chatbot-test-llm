import argparse
import yaml
import csv
import os
from analysis import analise_target
from testbench import target_from_object, evaluate_target, logger, enable_logging, LOG_DEBUG
from testbench._logging import INDENT, LOG_FLOAT_PRECISION

parser = argparse.ArgumentParser(description='LLM comparison for sentiment analysis in healthcare')
parser.add_argument('--data-file', type=str, default='./data/sentences.csv', help='input file path')
parser.add_argument('--bench-file', type=str, default='./data/bench.yml', help='benchmark configuration path')
enable_logging(level=LOG_DEBUG)

if __name__ == '__main__':
    args = parser.parse_args()
    with open(os.path.abspath(args.bench_file), 'r') as bench_definition_file:
        with open(os.path.abspath(args.data_file), 'r') as data_file:
            bench_list = yaml.load(bench_definition_file, Loader=yaml.FullLoader)
            reader = csv.reader(data_file)
            data = list(reader)
            targets = [target_from_object(bench) for bench in bench_list]
            logger.debug(f'Loaded {len(targets)} target' + ('s' if len(targets) > 1 else ''))
            reports = [evaluate_target(target, data) for target in targets]
            for report in reports:
                for question, responses in report:
                    logger.info(f'Question: {question}')
                    for response in responses:
                        logger.info(f'{INDENT}Output: {response.output}')
                        logger.info(f'{INDENT}Expected: {response.expected}')
                        logger.info(f'{INDENT}Correct: {response.correct}')
            statistics = [analise_target(target, data) for target in targets]
            for models_statistics in statistics:
                for model_statistics in models_statistics:
                    logger.info(f'{(model_statistics.accuracy * 100):.{LOG_FLOAT_PRECISION}f}% accuracy')


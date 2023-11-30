import argparse
import yaml
import csv
import os
from testbench import target_from_object, evaluate_target, logger, enable_logging, LOG_INFO
from testbench.logging import INDENT, LOG_FLOAT_PRECISION

parser = argparse.ArgumentParser(description='LLM comparison for sentiment analysis in healthcare')
parser.add_argument('--data-file', type=str, default='./data/general/test.csv', help='input file path')
parser.add_argument('--bench-file', type=str, default='./data/general/bench.yml', help='benchmark configuration path')

enable_logging(level=LOG_INFO)

if __name__ == '__main__':
    args = parser.parse_args()
    with open(os.path.abspath(args.bench_file), 'r') as bench_definition_file:
        with open(os.path.abspath(args.data_file), 'r') as data_file:
            logger.info(f'Loading data...')
            bench_list = yaml.load(bench_definition_file, Loader=yaml.FullLoader)
            reader = csv.reader(data_file)
            headers = next(reader)
            data = list(reader)
            targets = [target_from_object(bench) for bench in bench_list]
            logger.debug(f'Loaded {len(targets)} target' + ('s' if len(targets) > 1 else ''))
            reports = [evaluate_target(target, data) for target in targets]
            for report in reports:
                for question, responses in report:
                    logger.debug(f'Question: {question}')
                    for response in responses:
                        logger.debug(f'{INDENT}Output: {response.output}')
                        logger.debug(f'{INDENT}Expected: {response.expected}')
                        logger.debug(f'{INDENT}Correct: {response.correct}')

            # Bert score, each element is a dictionary with the following keys: precision, recall, f1
            scores = []
            logger.info('Computing BERT score...')
            for report in reports:
                for question, responses in report:
                    logger.debug(f'Question: {question}')
                    for response in responses:
                        bert_score = response.bert_comparison()
                        logger.debug(f'Score: {bert_score}')
                        scores.append(bert_score)
                # Compute average for each key
                precision = sum([score['precision'][0] for score in scores]) / len(scores)
                recall = sum([score['recall'][0] for score in scores]) / len(scores)
                f1 = sum([score['f1'][0] for score in scores]) / len(scores)
                logger.info(f'Results:'
                            f'\n\tPrecision: {precision:.{LOG_FLOAT_PRECISION}f}'
                            f'\n\tRecall: {recall:.{LOG_FLOAT_PRECISION}f}'
                            f'\n\tF1: {f1:.{LOG_FLOAT_PRECISION}f}')

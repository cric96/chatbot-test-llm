import argparse
import yaml
import csv
import os
import seaborn as sn
import matplotlib.pyplot as plt
from analysis import analise_request
from testbench import target_from_object, logger, enable_logging, LOG_DEBUG
from testbench._logging import LOG_FLOAT_PRECISION


parser = argparse.ArgumentParser(description='LLM comparison for sentiment analysis in healthcare')
parser.add_argument('--data-file', type=str, default='../data/request/request-test.csv', help='input file path')
parser.add_argument('--bench-file', type=str, default='../data/request/bench3params.yml', help='benchmark configuration path')
enable_logging(level=LOG_DEBUG)

if __name__ == '__main__':
    args = parser.parse_args()
    with open(os.path.abspath(args.bench_file), 'r') as bench_definition_file:
        with open(os.path.abspath(args.data_file), 'r') as data_file:
            bench_list = yaml.load(bench_definition_file, Loader=yaml.FullLoader)
            reader = csv.reader(data_file)
            headers = next(reader)
            data = list(reader)
            targets = [target_from_object(bench) for bench in bench_list]
            statistics = [analise_request(data) for target in targets]
            for models_statistics in statistics:
                for model_statistics in models_statistics:
                    logger.info(f'accuracy: {(model_statistics.accuracy):.{LOG_FLOAT_PRECISION}f}')
                    measure, quantity, format = model_statistics.one_by_one_accuracy
                    logger.info(f'measure: {measure:.{LOG_FLOAT_PRECISION}f}')
                    logger.info(f'quantity: {quantity:.{LOG_FLOAT_PRECISION}f}')
                    logger.info(f'format: {format:.{LOG_FLOAT_PRECISION}f}')
                    # plot confusion matrix and save it



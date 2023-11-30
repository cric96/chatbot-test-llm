import argparse
import yaml
import csv
import os
import seaborn as sn
import matplotlib.pyplot as plt
from analysis import analise_target
from testbench import target_from_object, evaluate_target, logger, enable_logging, LOG_DEBUG
from testbench._logging import INDENT, LOG_FLOAT_PRECISION

generate_plot = False
parser = argparse.ArgumentParser(description='LLM comparison for sentiment analysis in healthcare')
parser.add_argument('--data-file', type=str, default='./data/general/general-test.csv', help='input file path')
parser.add_argument('--bench-file', type=str, default='./data/general/bench.yml', help='benchmark configuration path')
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
            logger.debug(f'Loaded {len(targets)} target' + ('s' if len(targets) > 1 else ''))
            reports = [evaluate_target(target, data) for target in targets]
            for report in reports:
                for question, responses in report:
                    logger.info(f'Question: {question}')
                    for response in responses:
                        logger.info(f'{INDENT}Output: {response._output}')
                        logger.info(f'{INDENT}Expected: {response._expected}')
                        logger.info(f'{INDENT}Correct: {response.correct}')
            statistics = [analise_target(target, data) for target in targets]
            for models_statistics in statistics:
                for model_statistics in models_statistics:
                    logger.info('')
                    logger.info(f'accuracy: {(model_statistics.accuracy):.{LOG_FLOAT_PRECISION}f}')
                    measure, quantity, format = model_statistics.one_by_one_accuracy
                    logger.info(f'measure: {measure:.{LOG_FLOAT_PRECISION}f}')
                    logger.info(f'quantity: {quantity:.{LOG_FLOAT_PRECISION}f}')
                    logger.info(f'format: {format:.{LOG_FLOAT_PRECISION}f}')
                    if generate_plot:
                        # plot confusion matrix and save it
                        plt.figure()
                        sn.heatmap(model_statistics.confusion_matrix, annot=True, fmt='g')
                        plt.xlabel('Predicted')
                        plt.ylabel('Actual')
                        plt.savefig(f'confusion_matrix.png')

            ### bert score, each element is a dictionary with the following keys: precision, recall, f1
            scores = []
            for report in reports:
                for question, responses in report:
                    logger.info(f'Question: {question}')
                    for response in responses:
                        bert_score = response.bert_comparison()
                        logger.info(f'Score: {bert_score}')
                        scores.append(bert_score)
                ## Compute average for each key
                precision = sum([score['precision'][0] for score in scores]) / len(scores)
                recall = sum([score['recall'][0] for score in scores]) / len(scores)
                f1 = sum([score['f1'][0] for score in scores]) / len(scores)
                print(f'Precision: {precision}, Recall: {recall}, F1: {f1}')


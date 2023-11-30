import argparse
import pandas as pd
import yaml
import csv
import os
import seaborn as sn
import matplotlib.pyplot as plt
from analysis import analise_target
from testbench import target_from_object, evaluate_target, logger, enable_logging, LOG_DEBUG
from testbench._logging import INDENT, LOG_FLOAT_PRECISION

generate_plot = True
parser = argparse.ArgumentParser(description='LLM comparison for sentiment analysis in healthcare')
parser.add_argument('--data-file', type=str, default='./data/classification/sentences4-test.csv', help='input file path')
parser.add_argument('--bench-file', type=str, default='./data/classification/bench4.yml', help='benchmark configuration path')
enable_logging(level=LOG_DEBUG)

nas_bert_confusion_matrix = [[59, 3, 1, 0],
                             [1, 82, 0, 0],
                             [1, 0, 84, 0],
                             [1, 1, 1, 16]]

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
                        # logger.info(f'{INDENT}Correct: {response.correct}')
            statistics = [analise_target(target, data) for target in targets]
            for models_statistics in statistics:
                for model_statistics in models_statistics:
                    logger.info('')
                    accuracy = model_statistics.accuracy
                    logger.info(f'accuracy: {accuracy :.{LOG_FLOAT_PRECISION}f}')
                    cm = pd.DataFrame(nas_bert_confusion_matrix) # model_statistics.confusion_matrix
                    # columns and rows names into english
                    eng_names = ['General', 'Insertion', 'Request', 'Mood']
                    cm.columns, cm.index = eng_names, eng_names
                    # Add precision and recall
                    recalls = list(cm.apply(lambda row: row[row.name] / row.sum(), axis=0))
                    precisions = list(cm.apply(lambda col: col[col.name] / col.sum(), axis=1))
                    recalls.append(0)
                    precisions.append(0)

                    cm_with_pr = pd.DataFrame(columns=eng_names+['Recall'], index=eng_names+['Precision'], dtype=float)
                    cm_with_pr.iloc[0:4, 0:4] = cm
                    cm_with_pr['Recall'] = recalls
                    cm_with_pr.loc['Precision'] = precisions
                    logger.info(f'confusion matrix:\n {cm_with_pr}')
                    if generate_plot:
                        # plot confusion matrix and save it
                        sn.set(font_scale=1.4)
                        plt.figure()
                        sn.heatmap(cm, annot=True, fmt='g')
                        plt.xlabel('Predicted')
                        plt.ylabel('Actual')
                        plt.savefig(f'confusion_matrix.png', bbox_inches='tight')



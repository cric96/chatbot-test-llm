import argparse
import pandas as pd
import yaml
import csv
import os
import seaborn as sn
import matplotlib.pyplot as plt
from analysis import analise_target
from testbench import target_from_object, evaluate_target, logger, enable_logging, LOG_INFO
from testbench.logging import INDENT, LOG_FLOAT_PRECISION

parser = argparse.ArgumentParser(description='LLM comparison for sentiment analysis in healthcare')
parser.add_argument('--data-file', type=str, default='./data/classification/test.csv', help='input file path')
parser.add_argument('--bench-file', type=str, default='./data/classification/bench.yml',
                    help='benchmark configuration path')
enable_logging(level=LOG_INFO)

nas_bert_confusion_matrix = [[59, 3, 1, 0],
                             [1, 82, 0, 0],
                             [1, 0, 84, 0],
                             [1, 1, 1, 16]]


def plot_data(confusion_matrix: pd.DataFrame, model_name: str) -> None:
    # columns and rows names into english
    eng_names = ['General', 'Insertion', 'Request', 'Mood']
    confusion_matrix.columns, confusion_matrix.index = eng_names, eng_names
    # Add precision and recall
    recalls = list(confusion_matrix.apply(lambda row: row[row.name] / row.sum(), axis=0))
    precisions = list(confusion_matrix.apply(lambda col: col[col.name] / col.sum(), axis=1))
    recalls.append(0)
    precisions.append(0)

    cm_with_pr = pd.DataFrame(columns=eng_names + ['Recall'], index=eng_names + ['Precision'], dtype=float)
    cm_with_pr.iloc[0:4, 0:4] = confusion_matrix
    cm_with_pr['Recall'] = recalls
    cm_with_pr.loc['Precision'] = precisions
    logger.info(f'Confusion matrix for {model_name}:\n {cm_with_pr} \n')
    # plot confusion matrix and save it
    sn.set(font_scale=1.4)
    plt.figure()
    sn.heatmap(confusion_matrix, annot=True, fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'confusion_matrix_{model_name}.png', bbox_inches='tight')


if __name__ == '__main__':
    args = parser.parse_args()
    with open(os.path.abspath(args.data_file), 'r') as data_file:
        with open(os.path.abspath(args.bench_file), 'r') as bench_definition_file:
            logger.info(f'Loading data...')
            bench = target_from_object(yaml.load(bench_definition_file, Loader=yaml.FullLoader)[0])
            reader = csv.reader(data_file)
            headers = next(reader)
            data = list(reader)
            targets = [bench.new_target(target) for target in bench.models]
            logger.debug(f'Loaded {len(targets)} target' + ('s' if len(targets) > 1 else ''))
            reports = [evaluate_target(target, data, classification=True) for target in targets]
            for report in reports:
                for question, responses in report:
                    logger.debug(f'Question: {question}')
                    for response in responses:
                        logger.debug(f'{INDENT}Output: {response._output}')
                        logger.debug(f'{INDENT}Expected: {response._expected}')
            statistics = [analise_target(target, data, classification=True) for target in targets]
            for idx, models_statistics in enumerate(statistics):
                for model_statistics in models_statistics:
                    accuracy = model_statistics.accuracy
                    logger.info(f'Accuracy for {targets[idx].models[0]}: {accuracy :.{LOG_FLOAT_PRECISION}f}')
                    cm = model_statistics.confusion_matrix
                    plot_data(cm, targets[idx].models[0])

            # NAS-BERT
            cm = pd.DataFrame(nas_bert_confusion_matrix)
            logger.info(f'Accuracy for nas_bert: {cm.values.diagonal().sum() / cm.values.sum() :.{LOG_FLOAT_PRECISION}f}')
            plot_data(cm, 'nas_bert')

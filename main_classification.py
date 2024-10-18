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


LATEX_FLOAT_PRECISION = 2
parser = argparse.ArgumentParser(description='LLM comparison for sentiment analysis in healthcare')
parser.add_argument('--data-file', type=str, default='./data/classification/test.csv', help='input file path')
parser.add_argument('--bench-file', type=str, default='./data/classification/bench-gemini.yml',
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
    plt.title(model_name)
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
            cms = []
            for idx, models_statistics in enumerate(statistics):
                for model_statistics in models_statistics:
                    accuracy = model_statistics.accuracy
                    logger.info(f'Accuracy for {targets[idx].models[0]}: {accuracy :.{LOG_FLOAT_PRECISION}f}')
                    cm = model_statistics.confusion_matrix
                    cms.append(cm)
                    plot_data(cm, targets[idx].models[0])

            # NAS-BERT
            cm = pd.DataFrame(nas_bert_confusion_matrix)
            logger.info(f'Accuracy for nas_bert: {cm.values.diagonal().sum() / cm.values.sum() :.{LOG_FLOAT_PRECISION}f}')
            plot_data(cm, 'nas_bert')
            cms.append(cm)

            # Generate Latex table with precision and recall metrics for all 4 categories plus the overall accuracy
            with open('classification_test_results.tex', 'w') as f:
                f.write('\\begin{table*}[]\n\\centering\n\\resizebox{\\textwidth}{!}{\n')
                f.write('\\begin{tabular}{|l|rrrr|rrrr|r|}\n')
                f.write('\\hline\n')
                f.write('\\multicolumn{1}{|c|}{\\multirow{2}{*}{Model}} & \\multicolumn{4}{c|}{Precision} & \\multicolumn{4}{c|}{Recall} & \\multicolumn{1}{l|}{\\multirow{2}{*}{Accuracy}} \\\\ \\cline{2-9}\n')
                f.write('\\multicolumn{1}{|c|}{} & \\multicolumn{1}{l|}{General} & \\multicolumn{1}{l|}{Insertion} & \\multicolumn{1}{l|}{Request} & \\multicolumn{1}{l|}{Mood} & \\multicolumn{1}{l|}{General} & \\multicolumn{1}{l|}{Insertion} & \\multicolumn{1}{l|}{Request} & \\multicolumn{1}{l|}{Mood} & \\multicolumn{1}{l|}{} \\\\ \\hline\n')
                for idx, model in enumerate(targets):
                    if idx % 2 == 0:
                        f.write('\\rowcolor[HTML]{EFEFEF}\n')
                    f.write(f'{model.models[0].capitalize()} & ')
                    for i in range(4):
                        f.write('\\multicolumn{1}{r|}{' + f'{cms[idx].iloc[i, i] / cms[idx].iloc[i, :].sum() :.{LATEX_FLOAT_PRECISION}f}' + '} & ')
                    for i in range(4):
                        f.write('\\multicolumn{1}{r|}{' + f'{cms[idx].iloc[i, i] / cms[idx].iloc[:, i].sum() :.{LATEX_FLOAT_PRECISION}f}' + '} & ')
                    f.write(f'{cms[idx].values.diagonal().sum() / cms[idx].values.sum() :.{LATEX_FLOAT_PRECISION}f} \\\\ \\hline\n')
                if (idx + 1) % 2 == 0:
                    f.write('\\rowcolor[HTML]{EFEFEF}\n')
                f.write(f'ML.NET~2.0 framework & ')
                for i in range(4):
                    f.write('\\multicolumn{1}{r|}{' + f'{cms[-1].iloc[i, i] / cms[-1].iloc[i, :].sum() :.{LATEX_FLOAT_PRECISION}f}' + '} & ')
                for i in range(4):
                    f.write('\\multicolumn{1}{r|}{' + f'{cms[-1].iloc[i, i] / cms[-1].iloc[:, i].sum() :.{LATEX_FLOAT_PRECISION}f}' + '} & ')
                f.write(f'{cms[-1].values.diagonal().sum() / cms[-1].values.sum() :.{LATEX_FLOAT_PRECISION}f} \\\\ \\hline\n')
                f.write('\\end{tabular}\n')
                f.write('}\n\\caption{Results of the classification phase for all messages.\n%\n'
                        'Models used in the experiments are reported in the first column.\n%\n'
                        'The next two macro columns -- precision and recall -- report the corresponding metric per single class (general, insertion, request and mood).\n%\n'
                        'The last column shows the overall accuracy of the models.}\n\\label{categorization:results}\n\\end{table*}\n')
            logger.info('Classification test results saved to classification_test_results.tex')
            logger.info('Done')

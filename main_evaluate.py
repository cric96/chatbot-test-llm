import argparse
import yaml
import csv
import os
from testbench import target_from_object, evaluate_target, logger, enable_logging, LOG_INFO
from testbench.logging import INDENT, LOG_FLOAT_PRECISION


LATEX_FLOAT_PRECISION = 2
parser = argparse.ArgumentParser(description='LLM comparison for sentiment analysis in healthcare')
parser.add_argument('--data-file', type=str, default='./data/general/test.csv', help='input file path')
parser.add_argument('--bench-file', type=str, default='./data/general/bench.yml', help='benchmark configuration path')

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
            targets = [bench.new_target(target) for target in bench.models]
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
            meteor_scores = []
            logger.info('Computing BERT score...')
            precisions, recalls, f1s = [], [], []
            for idx, report in enumerate(reports):
                for question, responses in report:
                    logger.debug(f'Question: {question}')
                    for response in responses:
                        bert_score = response.bert_comparison()
                        meteor_score = response.meteor_comparison()
                        logger.debug(f'Score: {bert_score}')
                        scores.append(bert_score)
                        meteor_scores.append(meteor_score)
                # Compute average for each key
                precision = sum([score['precision'][0] for score in scores]) / len(scores)
                recall = sum([score['recall'][0] for score in scores]) / len(scores)
                f1 = sum([score['f1'][0] for score in scores]) / len(scores)
                # Compute average for meteor
                meteor = sum(meteor["meteor"] for meteor in meteor_scores) / len(meteor_scores)
                precisions.append(precision)
                recalls.append(recall)
                f1s.append(f1)
                logger.info(f'Results for {targets[idx].models[0]}:'
                            f'\n\tPrecision: {precision:.{LOG_FLOAT_PRECISION}f}'
                            f'\n\tRecall: {recall:.{LOG_FLOAT_PRECISION}f}'
                            f'\n\tF1: {f1:.{LOG_FLOAT_PRECISION}f}\n')
                logger.info(f'Meteor score for {targets[idx].models[0]}: {meteor:.{LOG_FLOAT_PRECISION}f}\n')

            # Generate Latex table with precision, recall and f1 for each model
            logger.info('Generating Latex table...')
            latex_table = '\\begin{table}[ht]\n\\centering\n\\begin{tabular}{|l|rrr|}\n\\hline\n'
            latex_table += '\\multicolumn{1}{|c|}{\\multirow{2}{*}{Model}} & \\multicolumn{3}{c|}{Bert Score} \\\\ \cline{2-4}\n'
            latex_table += '\\multicolumn{1}{|c|}{} & \\multicolumn{1}{c|}{Precision} & \\multicolumn{1}{c|}{Recall} & \\multicolumn{1}{c|}{F1} \\\\ \\hline\n'
            for idx, target in enumerate(targets):
                if idx % 2 == 0:
                    latex_table += '\\rowcolor[HTML]{EFEFEF}\n'
                latex_table += f'{target.models[0].capitalize()} & '
                latex_table += '\multicolumn{1}{r|} {' + f'{precisions[idx]:.{LATEX_FLOAT_PRECISION}f}' + '} & '
                latex_table += '\multicolumn{1}{r|} {' + f'{recalls[idx]:.{LATEX_FLOAT_PRECISION}f}' + '} & '
                latex_table += '\multicolumn{1}{r|} {' + f'{f1s[idx]:.{LATEX_FLOAT_PRECISION}f}' + '} \\\\ \\hline \n'
            latex_table += '\\end{tabular}\n'
            latex_table += '\\caption{Evaluation of LLMs responses via BERTScore: this analysis presents the outcomes of assessing open LLM\'s performances using BERTScore, based on a set of 128 questions and using as reference GPT-3.5.\n%\n'
            latex_table += 'The report includes average values of precision, F1 score, and recall as calculated through BERTScore metrics.}\n'
            latex_table += '\\label{bertscore}\n\\end{table}'
            with open('bert_score.tex', 'w') as latex_file:
                latex_file.write(latex_table)

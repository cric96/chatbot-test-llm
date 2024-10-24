import argparse
import yaml
import csv
import os
import pandas as pd
from testbench import target_from_object, evaluate_target, logger, enable_logging, LOG_INFO, update_cache_folder
from testbench.logging import INDENT, LOG_FLOAT_PRECISION


LATEX_FLOAT_PRECISION = 2
parser = argparse.ArgumentParser(description='LLM comparison for sentiment analysis in healthcare')
parser.add_argument('--data-file', type=str, default='./data/general/test-gemini.csv', help='input file path')
parser.add_argument('--bench-file', type=str, default='./data/general/bench.yml', help='benchmark configuration path')
parser.add_argument('--metrics-file', type=str, default='./data/general/metrics.yml', help='metrics configuration path')
parser.add_argument('--cache', type=str, default=None, help='cache directory path')

enable_logging(level=LOG_INFO)

if __name__ == '__main__':
    args = parser.parse_args()
    if args.cache is not None:
        update_cache_folder(args.cache)
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
            responses_all = {}
            for idx, report in enumerate(reports):
                responses_all[targets[idx].models[0]] = []
                for question, responses in report:
                    for response in responses:
                        responses_all[targets[idx].models[0]].append(response.output)
            # get all question for one report
            questions = [question for question, _ in reports[0]]
            ground_truth = [response.expected for _, responses in reports[0] for response in responses]
            # convert responses_all in a pd
            responses_df = pd.DataFrame(responses_all)
            # add questions and ground truth
            responses_df['question'] = questions
            responses_df['ground_truth'] = ground_truth
            # reorder columns with question and ground truth first
            responses_df = responses_df[['question', 'ground_truth'] + [col for col in responses_df.columns if col not in ['question', 'ground_truth']]]
            # store responses in a csv
            responses_df.to_csv('responses.csv', index=False)

            def evaluate_with_single_output_metrics(local_metrics: list[str], names: list[str]) -> dict[str: list[float]]:
                all_results: dict[str: list[float]] = {}
                for m in local_metrics:
                    all_results[m] = []
                for idx, report in enumerate(reports):
                    for question, responses in report:
                        logger.debug(f'Question: {question}')
                        for response in responses:
                            for m, name in zip(local_metrics, names):
                                if name != 'None':
                                    score = response.comparison(m)[name]
                                else:
                                    score = response.comparison(m)
                                logger.debug(f'{m} score: {score}')
                                all_results[m].append(score)
                for m in local_metrics:
                    if hasattr(all_results[m][0], '__iter__'):
                        all_results[m] = sum([x[0] for x in all_results[m]]) / len(all_results[m])
                    else:
                        all_results[m] = sum(all_results[m]) / len(all_results[m])
                    logger.info(f'{m} score: {all_results[m]:.{LOG_FLOAT_PRECISION}f}\n')
                return all_results

            def evaluate_with_multi_output_metric(metric: str, sub_metrics: list[str]) -> dict[str: list[float]]:
                all_results: dict[str: list[float]] = {}
                for sub_metric in sub_metrics:
                    all_results[sub_metric] = []
                for idx, report in enumerate(reports):
                    for question, responses in report:
                        logger.debug(f'Question: {question}')
                        for response in responses:
                            score = response.comparison(metric)
                            logger.debug(f'{metric} score: {score}')
                            for sub_metric in sub_metrics:
                                all_results[sub_metric].append(score[sub_metric])
                for sub_metric in sub_metrics:
                    if hasattr(all_results[sub_metric][0], '__iter__'):
                        all_results[sub_metric] = sum([x[0] for x in all_results[sub_metric]]) / len(all_results[sub_metric])
                    else:
                        all_results[sub_metric] = sum(all_results[sub_metric]) / len(all_results[sub_metric])
                    logger.info(f'{sub_metric} score: {all_results[sub_metric]:.{LOG_FLOAT_PRECISION}f}\n')
                return all_results

            def generate_latex_table_with_single_output_metrics(results: dict[str: list[float]]) -> str:
                latex_table = '\\begin{table}[ht]\n\\centering\n\\begin{tabular}{|l|c|}\n\\hline\n'
                latex_table += '\\multicolumn{1}{|c|} & '
                latex_table += ' &'.join(f'{{\\multicolumn{{1}}{{*}}{{{metric}}}}}' for metric in sorted(results.keys()))
                latex_table += ' \\\\ \\hline\n'
                for metric, score in results.items():
                    latex_table += f'{metric.capitalize()} & '
                    latex_table += '\multicolumn{1}{r|} {' + f'{score:.{LATEX_FLOAT_PRECISION}f}' + '} \\\\ \\hline \n'
                latex_table += '\\end{tabular}\n'
                caption = 'Evaluation of LLMs responses using '
                caption += ','.join(sorted(results.keys())).replace('_', ' ')
                caption += ' metrics. This analysis has been performed on a set of 128 questions and using as reference Gemini.\n%\n'
                latex_table += '\\label{tab:metrics}\n\\end{table}'
                return latex_table

            def generate_latex_table_with_multiple_output_metric(metric_name: str, results: dict[str: list[float]]) -> str:
                latex_table = '\\begin{table}[ht]\n\\centering\n\\begin{tabular}{|l|c|}\n\\hline\n'
                latex_table += '\\multicolumn{1}{|c|}{\\multirow{2}{*}{' + metric_name + '}} & '
                latex_table += ' &'.join(f'{{\\multicolumn{{1}}{{*}}{{{metric}}}}}' for metric in sorted(results.keys()))
                latex_table += ' \\\\ \\cline{2-' + str(len(results) + 1) + '}\n'
                for metric, score in results.items():
                    latex_table += f'{metric.capitalize()} & '
                    latex_table += '\multicolumn{1}{r|} {' + f'{score:.{LATEX_FLOAT_PRECISION}f}' + '} \\\\ \\hline \n'
                latex_table += '\\end{tabular}\n'
                caption = 'Evaluation of LLMs responses using '
                caption += ','.join(sorted(results.keys())).replace('_', ' ')
                caption += ' metrics. This analysis has been performed on a set of 128 questions and using as reference Gemini.\n%\n'
                latex_table += f'\\caption{{{caption}}}\n'
                latex_table += f'\\label{{tab:{metric_name}}}\n\\end{{table}}'
                return latex_table

            with open(os.path.abspath(args.metrics_file), 'r') as metrics_file:
                metrics = yaml.load(metrics_file, Loader=yaml.FullLoader)
                single_output_metrics = {list(x.keys())[0]: x[list(x.keys())[0]][0] for x in metrics['single_output']}
                multi_output_metrics = {list(x.keys())[0]: x[list(x.keys())[0]] for x in metrics['multi_output']}
                single_output_results = evaluate_with_single_output_metrics(list(single_output_metrics.keys()), list(single_output_metrics.values()))
                multi_output_results = {}
                for metric, sub_metrics in multi_output_metrics.items():
                    multi_output_results[metric] = evaluate_with_multi_output_metric(metric, sub_metrics)
                latex_table_single_output = generate_latex_table_with_single_output_metrics(single_output_results)
                with open('single_output_metrics.tex', 'w') as f:
                    f.write(latex_table_single_output)
                for metric, results in multi_output_results.items():
                    latex_table_multi_output = generate_latex_table_with_multiple_output_metric(metric, results)
                    with open(f'{metric}.tex', 'w') as f:
                        f.write(latex_table_multi_output)
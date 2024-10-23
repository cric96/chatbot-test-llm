import argparse
import yaml
import csv
import os
from analysis import analise_target, analise_request
from testbench import target_from_object, evaluate_target, logger, enable_logging, LOG_INFO, update_cache_folder
from testbench.logging import INDENT, LOG_FLOAT_PRECISION


parser = argparse.ArgumentParser(description='LLM comparison for sentiment analysis in healthcare')
parser.add_argument('--data-file', type=str, default='./data/request/test.csv', help='input file path')
parser.add_argument('--bench-file', type=str, default='./data/request/bench-full-prompt.yml', help='benchmark configuration path')
parser.add_argument('--cache', type=str, default="slm", help='cache directory path')

enable_logging(level=LOG_INFO)
LATEX_FLOAT_PRECISION = 2


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
            targets = [bench.new_target(model) for model in bench.models]
            logger.debug(f'Loaded {len(targets)} target' + ('s' if len(targets) > 1 else ''))
            reports = [evaluate_target(target, data) for target in targets]
            for report in reports:
                for question, responses in report:
                    logger.debug(f'Question: {question}')
                    for response in responses:
                        logger.debug(f'{INDENT}Output: {response._output}')
                        logger.debug(f'{INDENT}Expected: {response._expected}')
            # ChatGPT
            statistics = list(analise_request(data))
            result_values = {}
            for models_statistics in statistics:
                measure, quantity, format = models_statistics.one_by_one_accuracy
                result_values["ChatGPT3.5"] = [measure, quantity, format, models_statistics.accuracy]
                logger.info(f'Results: for ChatGPT3.5:'
                            f'\n\taccuracy: {models_statistics.accuracy:.{LOG_FLOAT_PRECISION}f}'
                            f'\n\tmeasure: {measure:.{LOG_FLOAT_PRECISION}f}'
                            f'\n\tquantity: {quantity:.{LOG_FLOAT_PRECISION}f}'
                            f'\n\tformat: {format:.{LOG_FLOAT_PRECISION}f}\n')

            statistics = [analise_target(target, data) for target in targets]
            for idx, models_statistics in enumerate(statistics):
                for model_statistics in models_statistics:
                    measure, quantity, format = model_statistics.one_by_one_accuracy
                    result_values[targets[idx].models[0]] = [measure, quantity, format, model_statistics.accuracy]
                    logger.info(f'Results: for {targets[idx].models[0]}:'
                                f'\n\taccuracy: {model_statistics.accuracy :.{LOG_FLOAT_PRECISION}f}'
                                f'\n\tmeasure: {measure:.{LOG_FLOAT_PRECISION}f}'
                                f'\n\tquantity: {quantity:.{LOG_FLOAT_PRECISION}f}'
                                f'\n\tformat: {format:.{LOG_FLOAT_PRECISION}f}\n')



        # Generate Latex table with accuracies for "measure", "quantity", "format" and the overall accuracy for each model
        # Sort alphabetically by model name
        result_values = {k: v for k, v in sorted(result_values.items(), key=lambda item: item[0])}
        with open('request_test_results.tex', 'w') as f:
            f.write('\\begin{table*}[]\n\\centering\n\\resizebox{\\textwidth}{!}{\n')
            f.write('\\begin{tabular}{|l|rrrr|}\n')
            f.write('\\hline\n')
            f.write('\\multicolumn{1}{|c|}{\\multirow{2}{*}{Model}} & \\multicolumn{4}{c|}{Accuracy} \\\\ \\cline{2-5}\n')
            f.write('\\multicolumn{1}{|c|}{} & \\multicolumn{1}{l|}{Measure} & \\multicolumn{1}{l|}{Quantity} & \\multicolumn{1}{l|}{Format} & \\multicolumn{1}{l|}{Overall} \\hline \n')
            for idx, (model_name, values) in enumerate(result_values.items()):
                if idx % 2 == 0:
                    f.write('\\rowcolor[HTML]{EFEFEF}\n')
                f.write(f'{model_name.capitalize()} & ')
                for i in range(4):
                    and_str = '} & ' if i < 3 else '} \\\\ \\hline\n'
                    f.write('\\multicolumn{1}{r|}{' + f'{values[i]:.{LATEX_FLOAT_PRECISION}f}' + and_str)
            f.write('\\end{tabular}\n')
            f.write('}\n\\caption{Results of the analysis phase for request messages.\n%\n'
                    'The first column describes the model used in the experiments (116 queries in total).\n%\n'
                    'The following four columns report the accuracy for the measure, the quantity, the format and for all combined.}\n\\label{tab:req}\n\\end{table*}\n')
        logger.info('request test results saved to request_test_results.tex')
        logger.info('Done')

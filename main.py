import argparse
import yaml
import os
from testbench import target_from_object

parser = argparse.ArgumentParser(description='LLM comparison for sentiment analysis in healthcare')
parser.add_argument('--data-file', type=str, default='./data/test.csv', help='input file path')
parser.add_argument('--bench-file', type=str, default='./data/bench.yml', help='benchmark configuration path')

if __name__ == '__main__':
    args = parser.parse_args()
    with open(os.path.abspath(args.bench_file), 'r') as f:
        bench = yaml.load(f, Loader=yaml.FullLoader)
        target = target_from_object(bench[0])
        reply = target.provider.use("llama2", target.system_prompt).ask("come una dieta sana può aiutare a gestire l’ipertensione?")
        print(reply)


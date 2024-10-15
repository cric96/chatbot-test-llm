import argparse

import pandas as pd
import yaml
import csv
import os

from gemini import GeminiService
from testbench import target_from_object, evaluate_target


parser = argparse.ArgumentParser(description='Data generation using an LLM as a ground truth')
parser.add_argument('--data-file', type=str, default='./data/general/test.csv', help='input file path')

prompt = """
Sei un chatbot che aiuta i pazienti a monitorare la loro pressione sanguigna e la loro frequenza cardiaca.
Rispondi alla domande dei clienti come se fossi un medico, in modo conciso e professionale. Rispondi SOLO in italiano.
"""
if __name__ == '__main__':
    service = GeminiService()
    model = service.use("gemini-1.5-pro-002", prompt)
    args = parser.parse_args()
    with open(os.path.abspath(args.data_file), 'r') as data_file:
        reader = csv.reader(data_file)
        headers = next(reader)
        data = list(reader)
        replies = []
        questions = []
        for row in data:
            question = row[0]
            questions.append(question)
            reply = model.ask(question)
            replies.append(reply)
            print(f"Question: {question}")
            print(f"Reply: {reply}")
            print("=====================================")
        df = pd.DataFrame(columns=headers)
        df['Response'] = replies
        df['Sentence'] = questions
        df.to_csv('output.csv', index=False)
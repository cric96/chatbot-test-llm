import argparse
import os

import pandas as pd
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase
from deepeval.test_case import LLMTestCaseParams

from eval import GoogleVertexAI

parser = argparse.ArgumentParser(description='LLM comparison for sentiment analysis in healthcare')
parser.add_argument('--data-file', type=str, default='./responses.csv', help='input file path')

result_folder = "data/llm_as_judge"
if __name__ == '__main__':
   args = parser.parse_args()
   df = pd.read_csv(args.data_file)
   to_judge = ["llama3.2:3b","gemma2:2b","phi3.5:latest", "qwen2.5:0.5b", "qwen2.5:1.5b", "qwen2.5:3b"]
   print(df[to_judge])
   model = GoogleVertexAI("gemini-1.5-pro-002", "GENAI_API_KEY")
   ground_truth = df["ground_truth"]

   # create a test case for each ground truth and columns
   for judging in to_judge:
      cases = []
      correctness_metric = GEval(
         name="Correctness of " + judging + " LLM",
         criteria="Determine whether the actual output is factually correct based on the expected output.",
         evaluation_steps=[
            "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
            "You should also heavily penalize omission of detail",
            "Since you are a chatbot which mimics a medical professional, you should also penalize any incorrect medical advice",
         ],
         evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
         model=model
      )
      for idx, row in df.iterrows():
         case = LLMTestCase(
            input=row["question"],
            actual_output=row[judging],
            expected_output=ground_truth[idx]
         )
         cases.append(case)
      dataset = EvaluationDataset(test_cases=cases, )
      result = dataset.evaluate([correctness_metric])
      print(f"HERE ++  {result.test_results}")
      if not os.path.exists(result_folder):
         os.makedirs(result_folder)
      with open(f"{result_folder}/{judging}.json", "w") as f:
         f.write(result.to_json())
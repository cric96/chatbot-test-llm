import argparse

import pandas as pd
import yaml
import csv
import os
from testbench import target_from_object, evaluate_target
import pandas as pd

parser = argparse.ArgumentParser(description='LLM comparison for sentiment analysis in healthcare')

ground_truths = {
    "human": "data/general/test-human.csv",
    "gemini": "data/general/test-gemini.csv",
}

base = "general_responses.csv"
if __name__ == '__main__':
    base = pd.read_csv(base)
    for key in ground_truths:
        ground_truth = pd.read_csv(ground_truths[key])
        print(ground_truth.columns)
        base["ground_truth_" + key] = ground_truth["Response"]
    base.to_csv("general_responses_ground_truth.csv", index=False)
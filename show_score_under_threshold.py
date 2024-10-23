import json
import numpy as np
names = [
    "phi3.5:latest",
    "qwen2.5:0.5b",
    "qwen2.5:1.5b",
    "qwen2.5:3b",
    "llama3.2:3b",
    "llama3.2:1b",
    "gemma2:2b",
    "ground_truth_gemini"
]

folder = "data/llm_as_judge"
results = {}
for name in names:
    with open(f"{folder}/{name}.json") as f:
        results[name] = json.load(f)

scores_map = {}
show_worst = 5
cut_at = 300
for name in names:
    scores = []
    tests_results = results[name]["test_results"]
    for test in tests_results:
        scores.append(test["metrics_data"][0]["score"])
    # zip scores with index
    scores_zip_index = list(zip(scores, range(len(scores))))
    # sort by score
    scores_zip_index.sort(key=lambda x: x[0])
    print("---- " + name + "----")
    for i in range(show_worst):
        print(f"Input: {tests_results[scores_zip_index[i][1]]['input']}")
        print(f"Actual output: {tests_results[scores_zip_index[i][1]]['actual_output'][:cut_at]}...")
        ## only first 100 characters
        print(f"Expected output: {tests_results[scores_zip_index[i][1]]['expected_output'][:cut_at]}...")
        ## reason
        print(f"Reason: {tests_results[scores_zip_index[i][1]]['metrics_data'][0]['reason']}")
        print(f"Score: {tests_results[scores_zip_index[i][1]]['metrics_data'][0]['score']}")
        print("____________________")
    print("vvvvvvvvvvvvvv")

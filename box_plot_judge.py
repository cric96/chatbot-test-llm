import json
import numpy as np
names = [
    "phi3.5:latest",
    "qwen2.5:0.5b",
    "qwen2.5:1.5b",
    "qwen2.5:3b",
    "llama3.2:3b",
    "gemma2:2b"
]

folder = "data/llm_as_judge"
results = {}
for name in names:
    with open(f"{folder}/{name}.json") as f:
        results[name] = json.load(f)

scores_map = {}
for name in names:
    scores = []
    for test in results[name]["test_results"]:
        scores.append(test["metrics_data"][0]["score"])
    scores_map[name] = scores

# plot a box plot, using seaborn
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.DataFrame(scores_map)
# wider figure
plt.figure(figsize=(10, 6))
sns.boxplot(data=df)

plt.show()
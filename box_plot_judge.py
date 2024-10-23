import json

import matplotlib.pyplot as plt
import pandas as pd
# plot a box plot, using seaborn
import seaborn as sns

names = [
    "phi3.5:latest",
    "qwen2.5:0.5b",
    "qwen2.5:1.5b",
    "qwen2.5:3b",
    "llama3.2:3b",
    "llama3.2:1b",
    "gemma2:2b",
    #"ground_truth_gemini"
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

df = pd.DataFrame(scores_map)
# wider figure
plt.figure(figsize=(10, 6))
# increase font (all)
plt.rcParams.update({'font.size': 14})
plt.tick_params(axis='x', labelrotation=45)
# put x label on 45 degree angle
sns.boxplot(data=df, notch=True)

plt.title("LLM scores (GEval)")
plt.ylabel("Score")
plt.tight_layout()
# save as pdf
#plt.show()
plt.savefig("box_plot_judge.pdf")
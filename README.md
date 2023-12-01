# Repository for the experiments of the paper "LLM-based Solutions for Healthcare Chatbots: a comparative analysis"

## Instruction to run the experiments

### Requirements

The experiments were run using Python 3.10, so it is recommended to use the same or more recent version.
The requirements are listed in the `requirements.txt` file. To install them, run the following command:

```bash
pip install -r requirements.txt
``` 

### Running the experiments

Because LLM are not deterministic, the results shown in the paper have been committed in the `testbench/cache` folder.
There are three different classes of experiments: 
- classification, classify the patient's message into one of the 4 classes: general, insertion, request, mood
- request: analyse a request message. The goal is to retrieve 3 parameters: the measure, the time window (i.e., numer of days, referred as quantity), the output format.
- general: evaluate a free text response to a general or mood message. We use ChatGPT3.5 responses and BERTScore to evaluate our results.

To run the experiments, run the following command:
```bash
python main_classification.py
```
```bash
python main_request.py
```
```bash
python main_general.py
```

All commands will output on the console the metrics for each model.
Moreover, the classification command will output the confusion matrix for each model.

### Generate new data
If you want to run yourself the experiments, you must first move the files in the cache elsewhere, and then run the same commands as before.
> Note 1: rerunning the experiments could generate different results (still similar) and the overall operation is time-consuming.

> Note 2: before running the experiments, you must download the ollama service in local or in one of your servers and modify the bench.yml files accordingly.

> Note 3: the language of the dataset is Italian.
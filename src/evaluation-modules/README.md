# Overview

This module aims to reorganize the silver data file as references and prediction results as hypotheses for evaluation purposes into a single file. 

## Installation

1. Navigate to the evaluation-modules folder:
```
cd evaluation-modules

```
2. Clone the GenerationEval repository:
```
git clone https://github.com/WebNLG/GenerationEval.git

```
* Details can be found in [GenerationEval WebNLG2020/README.md](https://github.com/WebNLG/GenerationEval)

3. Replace the default evaluation script:
You can either manually replace the eval.py file in the GenerationEval directory with the version in evaluation-modules, or use the provided shell script to automate the process.

```
# Option 1: Manually replace the file
cp eval.py GenerationEval/eval.py

# Option 2: Run the provided script
bash run-evaluation-pipeline.sh

```
## Preparing Evaluation Data

1. Convert Model Outputs to Evaluation Format:
Converts triples formatted of the model to evaluation format file. To do this task, run application in notebook form: [converting-to-evaluation-formatted.ipynb](https://github.com/u2018/ANTS/blob/main/evaluation-modules/converting-to-evaluation-formatted.ipynb)

Ensure the notebook accounts for different models (conve, conve_literale, gpt-4, gpt-3.5) and datasets (ESBM-DBpedia, FACES) by setting the system_ and dataset variables appropriately. This script should produce two files: refs.txt for references and hyp.txt for hypotheses.

2. Running the Evaluation

Use the eval.py script within the GenerationEval directory to evaluate the hypotheses against the references. The evaluation metrics include BLEU, METEOR, CHRF++, TER, BERT-based, and BLEURT scores.

```
# Navigate to the GenerationEval directory
cd GenerationEval

# Run the evaluation
python eval.py -R ../../data/ESBM-DBpedia/evaluation/conve_literale_gpt-4/refs.txt -H ../../data/ESBM-DBpedia/evaluation/conve_literale_gpt-4/hyp.txt -lng en -nr 1 -m bleu,meteor,chrf++,ter,bert,bleurt

```
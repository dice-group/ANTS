# ANTS: Knowledge Graph Abstractive Entity Summarization

![GitHub license](https://img.shields.io/github/license/dice-group/ANTS)
![GitHub stars](https://img.shields.io/github/stars/dice-group/ANTS?style=social)

Our approach aims to address the challenges of abstractive entity summarization in Knowledge Graphs (KGs) by generating optimal summaries that combine present triples with inferred missing (absent) triples using KG Embeddings (KGE) and Large Language Models (LLM) techniques.

<p align="center">
<img src="images/ANTs-new.jpg" width="75%">
</p>

## 🚀 Table of Contents
- [About the Project](#about-the-project)
- [Installation](#installation)
- [Usage](#usage)
- [Example](#example)
- [Results](#results)
- [Contribution](#contribution)
- [License](#license)

---
## 📌 About the Project
ANTS generates entity summaries in natural language from Knowledge Graphs by leveraging both KGE and LLM techniques. It addresses the problem of missing information by predicting absent triples and verbalizing them into readable summaries.

---
## ⚙️ Installation
To run the ANTS framework, you need to install the following packages:

```bash
python 3.7+
torch
```

1. Create and activate a Conda environment:
```bash
conda create --name ants python=3.7
conda activate ants
```
2. Download the project
```bash
git clone https://github.com/dice-group/ANTS.git

# Navigate to ANTS directory
cd ANTS
```

2. Install required packages:
```bash
pip install torch
pip install -r requirements.txt
```

> ⚠️ **Important Note:** Ensure that all dependencies are correctly installed.

---
## 📂 Repository Structure
```
├── data
│   ├── ESBM-DBpedia
│   │   ├── ESSUM
│   │   │   ├── silver-standard-summaries
│   │   │   └── absent
│   │   ├── predictions
│   │   │   ├── ANTS
│   │   │   ├── baselines
│   │   │   ├── KGE
│   │   │   └── LLM
│   │   └── elist.txt
│   └── FACES
│       ├── ESSUM
│       │   ├── silver-standard-summaries
│       │   └── absent
│       ├── predictions
│       │   ├── ANTS
│       │   ├── baselines
│       │   ├── KGE
│       │   └── LLM
│       └── elist.txt
├── evaluation-modules
├── KGE-triples
├── LLM-triples
├── ranking-modules
├── verbalizing-modules
├── LICENSE
└── README.md
```
## 📊 Data Preparation

### ESSUM Dataset

A silver-standard dataset combining entities from ESBM-DBpedia and FACES. For each entity, we extract sentences with mentioned entities from the first paragraph of its Wikipedia page. In our experiment, we created two subsets: (1) ESSUM-DBpedia: 110 entities from ESBM-DBpedia, and (2) ESSUM-FACES: 50 entities from FACES.

<p align="center">
<img src="images/silver-summary-example-alt2.jpg" width="75%">
</p>

### ESSUM-ABSENT

Derived by randomly removing 20% of triples from ESBM-DBpedia and FACES. These omitted triples serve as ground-truth absent triples to evaluate a model’s ability to infer missing facts.


---
## 🛠️ Usage
### **1️⃣ KGE-Triples**
```bash
# Clone the LiteralE repository
git clone https://github.com/SmartDataAnalytics/LiteralE.git

# Navigate to the LiteralE directory and download the DBpedia dataset
cd LiteralE/data
wget https://zenodo.org/records/10991461/files/dbpedia34k.tar.gz
tar -xvf dbpedia34k.tar.gz

# Execute the script for missing triples prediction
python run_missing_triples_prediction.py dataset dbpedia34k model Conve_text input_drop 0.2 embedding_dim 100 batch_size 1 epochs 100 lr 0.001 process True
```

### **2️⃣ LLM-Triples**
This component leverages a Large Language Model (LLM), such as GPT, to extend its application to knowledge graph (KG) completion tasks, including triple classification, relation prediction, and the completion of missing triples. As illustrated below, the ANTS approach integrates the LLM-triples component, such as GPT-4, to address the inherent limitations of KGE methods in inferring literal triples.

<p align="center">
<img src="images/prompt-ants2.jpg" width="75%">
</p>

```bash
cd LLM-triples
python run_missing_triples_prediction.py
```
### **3️⃣ Triple-Ranking And Entity Summary**
Triples ranking utilizes the frequency of predicate occurrences within the knowledge graph, such as DBpedia. Predicates that occur most frequently will prioritize their corresponding triples at the top of the list. Run the ```triples-ranking``` process (which includes the ranking process and entity summary).

```
# Navigate to ranking-modules directory
cd ranking-modules

# Run triple-ranking and entity summary
python triples-ranking.py  --kge_model conve_text --llm_model gpt-4 --combined_model conve_text_gpt-4 --dataset ESBM-DBpedia --base_model ANTS
```
---
## How to Cite
```bibtex
@inproceedings{ANTS2025,
  author = {Firmansyah, Asep Fajar and Zahera, Hamada and Sherif, Mohamed Ahmed and and Moussallem, Diego and Ngonga Ngomo, Axel-Cyrille},
  booktitle = {ESWC2025},
  title = {ANTS: Abstractive Entity Summarization in Knowledge Graphs},
  year = 2025
}
```
---
## Contact
If you have any questions or feedbacks, feel free to contact us at asep.fajar.firmansyah@upb.de


# ANTS: Knowledge Graph Abstractive Entity Summarization

![GitHub license](https://img.shields.io/github/license/u2018/ANTS)
![GitHub stars](https://img.shields.io/github/stars/u2018/ANTS?style=social)

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
│   │   │   ├── LLM
│   │   │   └── ESSUM-Absent
│   │   └── elist.txt
│   └── FACES
│       ├── ESSUM
│       │   ├── silver-standard-summaries
│       │   └── absent
│       ├── predictions
│       │   ├── ANTS
│       │   ├── baselines
│       │   ├── KGE
│       │   ├── LLM
│       │   └── ESSUM-Absent
│       └── elist.txt
├── evaluation-modules
├── KGE-triples
├── LLM-triples
├── ranking-modules
├── verbalizing-modules
├── LICENSE
└── README.md
```

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

---
## 📊 Results
| Entity  | Gold Summary | Generated Summary |
|---------|-------------|-------------------|
| **Barack Obama** | Former US President | Barack Obama was born in Hawaii and served as the 44th President of the USA. |

---
## 🤝 Contribution Guide
1. 🍴 **Fork** the repository.
2. Clone your forked repo:
   ```bash
   git clone https://github.com/your-username/ANTS.git
   ```
3. Create a new branch:
   ```bash
   git checkout -b feature-new
   ```
4. Make your changes and commit:
   ```bash
   git commit -m "Added new feature"
   ```
5. Push and create a **pull request** 🚀

---
## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
### 🌟 Show your support!
If you like this project, consider **starring** 🌟 it on GitHub!

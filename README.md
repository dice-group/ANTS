# ANTS: Knowledge Graph Abstractive Entity Summarization

Our approach aims to address the challenges of abstractive entity summarization in Knowledge Graphs (KGs) by generating optimal summaries that combine present triples with inferred missing (absent) triples using KG Embeddings (KGE) and Large Language Models (LLM) techniques.

<p align="center">
<img src="images/ANTs.jpg" width="60%">
</p>

## Repository Structure: 
```
├── data
│   ├── essum                # Contains benchmark datasets
│   │   ├── dbpedia
│   │   └── faces
│   └── evaluation-results   # Contains the experiment results evaluated on the benchmark
│       ├── dbpedia
│       │   ├── baselines    # Contains baseline experiment results including GATES, DeepLENS, etc.
│       │   └── ants         # Contains results of our approach
│       └── faces
│           ├── baselines    # Contains baseline experiment results including GATES, DeepLENS, etc.
│           └── ants         # Contains results of our approach
├── LICENSE
└── README.md

```
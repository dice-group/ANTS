# ANTS: Knowledge Graph Abstractive Entity Summarization

Our approach aims to address the challenges of abstractive entity summarization in Knowledge Graphs (KGs) by generating optimal summaries that combine present triples with inferred missing (absent) triples using KG Embeddings (KGE) and Large Language Models (LLM) techniques.

<p align="center">
<img src="images/ANTs.jpg" width="60%">
</p>

## Repository Structure: 
```
├── data
│   └── essum
|        ├── dbpedia
|        └── faces
├── LICENSE
└── README.md
```

## Data preparation
### Create Essum Dataset
The Essum dataset consists of essum-dbpedia and essum-faces, which are derived from ESBM (version 1.2) and FACES datasets, respectively. They are designed to evaluate our approach, which combines absent triples with present triples to construct entity summaries. The dataset is created through the following procedures:

1. **Preparing the dataset**: the dataset is loaded from ESBM-DBpedia or FACES
   - ESBM-DBpedia contains 125 entities with 4436 triples.
   - FACES is composed of 50 entities with 2152 triples
3. **Removing 20% of triples**: The removed percentage of triples is calculated from 20% of the number of triples in each entity description, and it randomly selects that many triples to remove.
4. **Saving the Updated Dataset:**: The removed triples will be stored as the test set, while the remaining triples will be divided into training and validation sets with a 90:10 ratio.
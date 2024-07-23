# ESSUM
ESSUM dataset is derived from the ESBM (version 1.2) and FACES datasets and is designed to evaluate the abstractive entity summarization model. This model combines absent triples with present triples to construct abstractive entity summaries.

ESSUM dataset is composed of:
1. ESSUM-Triples
2. ESSUM-Abstract-Summaries

## Creation of the ESSUM-Triples Dataset
The ESSUM-Triples dataset is designed to evaluate whether baseline models, knowledge graph embedding (KGE) models, and large language models (LLM) can predict or infer absent triples in their generated entity summaries by selecting top-1o triples summaries.

The ESSUM-Triples dataset is created through the following procedures:

1. **Preparation of the Dataset from ESBM-DBpedia (version 1.2) and FACES:** 

   We selected 125 entities from ESBM-DBpedia and 50 entities from FACES. Each entity in these datasets consists of triples that represent the entity. Overall, the datasets comprise 4436 triples from ESBM-DBpedia and 2152 triples from FACES.  

2. **Removing 20% of Triples from Each Entity Description:**

    We assume that the original datasets (ESBM-DBpedia and FACES) have complete triples for each KG entity, representing present triples. To evaluate the ability of existing entity summarization models to predict absent triples, we remove 20% of the triples from each entity description through automatic random selection. The removed triples are considered absent triples.

3. **Dividing the Triples into Training, Validation, and Test Sets:**

    The removed triples are stored as the test set, while the remaining triples are divided into training and validation sets with a 90:10 ratio.

## Creation of the ESSUM-Abstract-Summaries Dataset
The ESSUM-Abstract-Summaries dataset is aimed at benchmarking the performance of abstractive summarization models on KGs.

The ESSUM-Abstract-Summaries dataset is created through the following procedures:

1. **Preparation of the Dataset:**
   We select the entities from the ESBM-DBpedia and FACES datasets that exist in Wikipedia articles. Out of the 125 entities from ESBM-DBpedia, 110 are found in Wikipedia articles and contain at least one paragraph of information.
3. **Retrieving Abstract-Summaries:**
   ESSUM-Abstract-Summaries are created by selecting sentences from Wikipedia articles that mention named entities relevant to the subject. Specifically, the first two paragraphs of the Wikipedia articles are retrieved to ensure the inclusion of relevant information. 
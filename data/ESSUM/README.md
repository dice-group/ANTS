# ESSUM
The ESSUM dataset is derived from the ESBM (version 1.2) and FACES datasets and is designed to evaluate the abstractive entity summarization model. This model combines absent triples with present triples to construct abstractive entity summaries.

## Creation of the ESSUM Dataset
The ESSUM dataset is created through the following procedures:

1. **Preparation of the Dataset from ESBM-DBpedia (version 1.2) and FACES:** 

   We selected 125 entities from ESBM-DBpedia and 50 entities from FACES. Each entity in these datasets consists of triples that represent the entity. Overall, the datasets comprise 4436 triples from ESBM-DBpedia and 2152 triples from FACES.  

2. **Removing 20% of Triples from Each Entity Description:**

    We assume that the original datasets (ESBM-DBpedia and FACES) have complete triples for each KG entity, representing present triples. To evaluate the ability of existing entity summarization models to predict absent triples, we remove 20% of the triples from each entity description through automatic random selection. The removed triples are considered absent triples.

3. **Dividing the Triples into Training, Validation, and Test Sets:**

    The removed triples are stored as the test set, while the remaining triples are divided into training and validation sets with a 90:10 ratio.
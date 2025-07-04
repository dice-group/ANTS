## 📦 ESSUM (Silver-Standard Summaries) — Updated Version

We release an improved version of the **ESSUM-DBpedia** and **ESSUM-FACES** datasets, part of the ESSUM benchmark for evaluating **abstractive entity summarization**.

### 🔧 What's New?

This update resolves a technical issue found in the earlier version, where **some entity descriptions contained more than one paragraph** from their corresponding Wikipedia articles. This was caused by a bug in the extraction script.

### ✅ What’s Fixed?

- Each entity is now paired with **only the first paragraph** of its Wikipedia article, as originally intended.
- The correction ensures **more accurate, consistent, and concise** silver-standard summaries across the dataset.

### 📌 Why It Matters

The improved dataset better aligns with the benchmark’s objective of evaluating models on generating **focused and informative entity summaries**. This is particularly important in domains such as **DBpedia** and **FACES**, where entity descriptions may be verbose or diverse in content.

### 📢 Recommendation

We strongly recommend using this updated version for future experiments to ensure:
- Reliable and consistent evaluation results  
- Fair comparisons across models  
- Proper alignment with the evaluation setup defined in the ESSUM benchmark  

---

## 📊 Evaluation Results on ESSUM Benchmark

| System         | BLEU | METEOR | chrF++ | TER  | BLEURT |
|----------------|------|--------|--------|------|--------|
| **ESSUM_DBpedia** |      |        |        |      |        |
| ESA            | 4.04 | 0.13   | 0.28   | 2.05 | 0.36   |
| AutoSUM        | 4.71 | 0.14   | 0.28   | 2.35 | 0.36   |
| DeepLENS       | 4.80 | 0.14   | 0.29   | 1.90 | 0.37   |
| GATES          | 4.63 | 0.14   | 0.28   | 1.83 | 0.37   |
| **ANTS (ours)**| **5.87** | **0.15** | **0.30** | **1.69** | **0.37** |

| **ESSUM_FACES**    |      |        |        |      |        |
| ESA            | 3.58 | 0.10   | 0.21   | **1.13** | 0.34   |
| AutoSUM        | 4.43 | 0.10   | 0.22   | 1.21 | 0.35   |
| DeepLENS       | 3.15 | 0.09   | 0.20   | 1.13 | 0.33   |
| GATES          | 4.05 | 0.10   | 0.22   | 1.22 | 0.34   |
| **ANTS (ours)**| **5.57** | **0.13** | **0.25** | 1.17 | **0.39** |

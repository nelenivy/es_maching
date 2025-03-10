# es_maching
Matching algorithms for event sequences

# datasets
MBD, data fusion, cikm2016

MBD - dataset introduced in https://arxiv.org/html/2409.17587v1
MBD and data fusion datasets are processed in https://github.com/Dzhambo/MBD
How to get data is listed in the repository

cikm2016 - dataset introduced in https://competitions.codalab.org/competitions/11171
data files are downloaded from https://drive.google.com/drive/folders/0B7XZSACQf0KdNXVIUXEyVGlBZnc
the dataset is converted to ptls format by notebooks

# approaches

# Retriever and Reranker Algorithm

This algorithm uses a two-stage approach: Retriever + Reranker, for efficient and accurate information retrieval.

## Stage 1: Retriever

The Retriever stage focuses on quickly finding the nearest neighbors based on embeddings. It employs a late fusion technique using a dual-encoder architecture.

*   **Technique:** Late Fusion
*   **Architecture:** Dual-encoder
*   **Similarity Function:** `f_sim(q, d) = <emb_q, emb_d>` where `emb_q` and `emb_d` are embeddings of the query (q) and document (d) respectively, and `<>` denotes the dot product.

## Stage 2: Reranker

The Reranker stage refines the results from the Retriever by performing a more thorough, but computationally intensive, search over the candidate documents. It utilizes middle and late fusion techniques.

*   **Techniques:** Middle Fusion, Late Fusion
*   **Architecture:** Dual-encoder
*   **Similarity Function (Middle Fusion):** `f_sim(q, d) = MLP(emb_q, emb_d)` where `MLP` is a multi-layer perceptron that takes the query and document embeddings as input.

## Fusion Types

*   **Late Fusion:**  Combines the results of independent encoding processes.
*   **Middle Fusion:**  Combines the query and document embeddings before the final similarity calculation.
*   **Early Fusion:** (End-to-end training with Cross-encoder) Events in the query (q) and document (d) are ordered chronologically.

![alt text](image.png)

## Results

| Algorithm                                         | F1 Score | Precision | Recall |
|---------------------------------------------------|----------|-----------|--------|
| Late fusion, RNN                                  | 0.39     | 0.39      | 0.39   |
| Middle fusion, RNN                                | 0.44     | 0.52      | 0.37   |
| Middle fusion, best from articles (TGCE)           | 0.46     | 0.49      | 0.44   |
| Middle fusion, transformers                       | 0.47     | 0.57      | 0.4    |
| Early fusion, best from articles                  | 0.51     | 0.57      | 0.58   |
| Early fusion + blending, modality token concat, transformers | 0.545    | 0.7       | 0.44   |

# bash files for running training 
bash files with launching runs are .sh files in the root repo.
files with prefix train_data_fusion are for training on data fusion dataset
files with prefix train_MBD are for training on MBD dataset
other bash files are for training on cikm2016 datasets



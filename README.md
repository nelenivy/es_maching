# es_matching
Matching algorithms for event sequences.

# Algorithm description

This algorithm uses a two-stage approach: Retriever + Reranker, for efficient and accurate information retrieval.

![alt text](image.png)
## Stage 1: Retriever

The Retriever stage focuses on quickly finding the nearest neighbors based on embeddings. It employs a late fusion technique using a dual-encoder architecture.

*   **Technique:** Late Fusion
*   **Architecture:** Dual-encoder
*   **Similarity Function:** `f_sim(q, d) = <emb_q, emb_d>` where `emb_q` and `emb_d` are embeddings of the query (q) and document (d) respectively, and `<>` denotes the dot product.

## Stage 2: Reranker

The Reranker stage refines the results from the Retriever by performing a more thorough, but computationally intensive, search over the candidate documents. It utilizes middle and late fusion techniques.

*   **Techniques:** Middle Fusion, Early Fusion
*   **Architecture:** Dual-encoder
*   **Similarity Function (Middle Fusion):** `f_sim(q, d) = MLP(emb_q, emb_d)` where `MLP` is a multi-layer perceptron that takes the query and document embeddings as input.

## Fusion Types

![alt text](image-1.png)

*   **Late Fusion:**  Combines the results of independent encoding processes.
*   **Middle Fusion:**  Combines the query and document embeddings before the final similarity calculation.
*   **Early Fusion:** (End-to-end training with Cross-encoder) Events in the query (q) and document (d) are ordered chronologically.


## Datasets

This project utilizes several datasets for training and evaluation:

*   **MBD:** The MBD dataset is described in detail in the paper available at [arXiv:2409.17587v1](https://arxiv.org/html/2409.17587v1).

*   **Data Fusion:** The MBD and Data Fusion datasets are preprocessed and available in the [Dzhambo/MBD](https://github.com/Dzhambo/MBD) repository.  Refer to the repository for instructions on accessing the data.

*   **CIKM2016:** This dataset was introduced in the CIKM Cup 2016 competition.

    *   Competition Link: [CIKM Cup 2016](https://competitions.codalab.org/competitions/11171)
    *   Data Download: The data files can be downloaded from this [Google Drive folder](https://drive.google.com/drive/folders/0B7XZSACQf0KdNXVIUXEyVGlBZnc).
    *   Data Conversion: Notebooks are used to convert the dataset into the ptls format.

## Results

# cikm2016 dataset

| Algorithm                                         | F1 Score | Precision | Recall |
|---------------------------------------------------|----------|-----------|--------|
| Late fusion, RNN                                  | 0.39     | 0.39      | 0.39   |
| Middle fusion, RNN                                | 0.44     | 0.52      | 0.37   |
| Middle fusion, best from articles (TGCE)           | 0.46     | 0.49      | 0.44   |
| Middle fusion, transformers                       | 0.47     | 0.57      | 0.4    |
| Early fusion, best from articles                  | 0.51     | 0.57      | 0.58   |
| Early fusion + blending, modality token concat, transformers | 0.545    | 0.7       | 0.44   |

# MBD dataset

## Retriever, Augmentation Ablation Results (Recall@100)

*   **Batch Size:** Train/Valid: 128
*   **Head:** Input Size 128, 2x128 hidden layers, Regression.
*   **Loss:** Matching Softmax Loss.
*   **Optimizer:** AdamW, LR 0.0001, Weight Decay 1e-5.
*   **LR Scheduler:** StepLR, Step Size 1, Gamma 0.95.
*   **Warmup Steps:** 2000.
*   **Epochs:** 15

| Configuration                                                        | geo2trx (valid/r2b) | trx2geo (valid/b2r) |
| :------------------------------------------------------------------- | :------------------ | :------------------ |
| COLES Augmentation                                                  | 0.307343            | 0.311181            |
| No Augmentation                                                     | 0.307786            | 0.313838            |
| No Augmentation, Batch Size 256                                     | 0.303653            | 0.309410            |
| COLES Augmentation, Loss Only for Matching Different Modalities | 0.324945            | 0.325092            |

## Reranker Model Performance

*   **Batch Size:** Train 1024
*   **Loss:** BCEWithLogitsLoss
*   **Optimizer:** AdamW, LR 1e-4, Weight Decay 1e-5
*   **LR Scheduler:** StepLR, Step Size 1, Gamma 0.95
*   **Warmup Steps:** 200
| Model                                                                 | ROC AUC | F1    |
| :-------------------------------------------------------------------- | :------ | :---- |
| Vanilla Cross Encoder                                                 | 0.712   | 0.665 |
| Early Fusion + Blending                                               | 0.716   | 0.668 |
| Early Fusion + Blending + Dynamic Time Bias                          | 0.718   | 0.668 |
| Early Fusion + Blending + Dynamic (Time + Modalities) Bias | 0.717   | 0.670 |


# bash files for running training 
bash files with launching runs are .sh files in the root repo.

files with prefix train_data_fusion are for training on data fusion dataset

files with prefix train_MBD are for training on MBD dataset

other bash files are for training on cikm2016 datasets



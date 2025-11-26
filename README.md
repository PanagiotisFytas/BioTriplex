# BioTriplex

[//]: # (Baseline Replication for BioTriplex - Biomedical Relation Extraction Dataset)

This repository contains the code and instructions to replicate the baseline results for the BioTriplex dataset,
a biomedical Relation Extraction (RE) and Named Entity Recognition (NER) dataset. The dataset includes expert 
annotations on PubMedCentral full-text articles, focusing on Gene, Disease and the relations between them.

This repo provides:

- the **BioTriplex corpus** (`/data/BioTriplex Corpus`) (full-text gene, disease and relation annotations),
- preprocessed splits for model training (`/data/Preprocessed BioTriplex`),
- BERT-based baselines, and
- the LLaMA-based experiments, via the llama-rec framework.

## Repository Structure

```
BioTriplex-results/
├── LICENSE                     # Code license (MIT)
├── README.md                   # This file
├── code
│   ├── README.md               # Additional code-level documentation (if provided)
│   ├── bert                    # BERT-based RE baselines on BioTriplex
│   │   ├── __init__.py
│   │   ├── biotriplex_qakshot_dataset.py
│   │   └── finetune_biotriplex_bert.py
│   └── llama-rec               # LLaMA-based RE experiments (llama-rec framework)
│       ├── CODE_OF_CONDUCT.md
│       ├── CONTRIBUTING.md
│       ├── README.md
│       ├── UPDATES.md
│       ├── dev_requirements.txt
│       ├── docs
│       ├── pyproject.toml
│       ├── recipes
│       ├── requirements.txt
│       ├── src
│       └── tools
└── data
    ├── BioTriplex Corpus   # The annotated BioTriplex full-text articles
    │   ├── 2275485
    │   ├── 2807459
    │   ├── ...
    │   └── 9992796
    ├── README.md              # Dataset description / format details
    └── Preprocessed BioTriplex   # Train/val/test splits for modelling
        ├── splits.json
        ├── test_para.txt
        ├── test_shorter.txt
        ├── train_para.txt
        ├── train_shorter.txt
        ├── val_para.txt
        └── val_shorter.txt
```

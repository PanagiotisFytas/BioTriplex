##!/usr/bin/env bash


# for general relations with no negative samples and threshold optimization
python bert/finetune_biotriplex_bert.py \
  --model_name microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext \
  --data_path "../../data/Preprocessed BioTriplex" \
  --output_dir "output_dir/" \
  --epochs 30 \
  --batch_size 16 \
  --lr 1e-4 \
  --weight_decay 0.01 \
  --max_length 512 \
  --group_relations \
  --wandb \
  --no_neg \
  --optimize_threshold \
  --wandb_run_name pubmedbert_biotriplex_model_no_neg_opt_genrel \
  --general_relations \
  --wandb_project biotriplex-bert

# for full relations with no negative samples and threshold optimization
python bert/finetune_biotriplex_bert.py \
  --model_name microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext \
  --data_path "../../data/Preprocessed BioTriplex" \
  --output_dir "output_dir/" \
  --epochs 30 \
  --batch_size 16 \
  --lr 1e-4 \
  --weight_decay 0.01 \
  --max_length 512 \
  --group_relations \
  --wandb \
  --no_neg \
  --optimize_threshold \
  --wandb_run_name pubmedbert_biotriplex_model_no_neg_optl \
  --wandb_project biotriplex-bert

# for full relations without threshold optimization
python bert/finetune_biotriplex_bert.py \
  --model_name microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext \
  --data_path "../../data/Preprocessed BioTriplex" \
  --output_dir "output_dir/" \
  --epochs 30 \
  --batch_size 16 \
  --lr 1e-4 \
  --weight_decay 0.01 \
  --max_length 512 \
  --group_relations \
  --wandb \
  --no_neg \  --optimize_threshold \
  --wandb_run_name pubmedbert_biotriplex_model_no_neg \
  --wandb_project biotriplex-bert

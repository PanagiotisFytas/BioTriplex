#!/bin/bash

# for biobert with optimized thresholding, upweighting, no negative samples, and general relations
python bert/finetune_biotriplex_bert.py \
  --model_name dmis-lab/biobert-v1.1 \
  --data_path "../../data/Preprocessed BioTriplex" \
  --output_dir "output_dir" \
  --epochs 30 \
  --batch_size 16 \
  --lr 1e-4 \
  --weight_decay 0.01 \
  --max_length 512 \
  --group_relations \
  --wandb \
  --no_neg \
  --upweight \
  --optimize_threshold \
  --wandb_run_name biobert_biotriplex_model_upweight_no_neg_opt_genrel \
  --general_relations \
  --wandb_project biotriplex-bert

# for biobert with optimized thresholding, upweighting, no negative samples, and regular relations
python bert/finetune_biotriplex_bert.py \
  --model_name dmis-lab/biobert-v1.1 \
  --data_path "../../data/Preprocessed BioTriplex" \
  --output_dir "output_dir" \
  --epochs 30 \
  --batch_size 16 \
  --lr 1e-4 \
  --weight_decay 0.01 \
  --max_length 512 \
  --group_relations \
  --wandb \
  --no_neg \
  --upweight \
  --optimize_threshold \
  --wandb_run_name biobert_biotriplex_model_upweight_no_neg_opt \
  --wandb_project biotriplex-bert

# for biobert without optimized thresholding, upweighting, no negative samples, and regular relations
python bert/finetune_biotriplex_bert.py \
  --model_name dmis-lab/biobert-v1.1 \
  --data_path "../../data/Preprocessed BioTriplex" \
  --output_dir "output_dir" \
  --epochs 30 \
  --batch_size 16 \
  --lr 1e-4 \
  --weight_decay 0.01 \
  --max_length 512 \
  --group_relations \
  --wandb \
  --no_neg \
  --upweight \
  --wandb_run_name biobert_biotriplex_model_upweight_no_neg \
  --wandb_project biotriplex-bert
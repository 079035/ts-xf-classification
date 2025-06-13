#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# Base configuration for imbalanced classification
model_name=iTransformer
data_path=data  # Folder containing train.parquet, val.parquet, test.parquet

# Experiment 1: Focal Loss with default parameters
python -u run_imbalanced.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./data/ \
  --data_path $data_path \
  --model_id imbalanced_focal \
  --model $model_name \
  --data custom \
  --label_col label \
  --e_layers 3 \
  --batch_size 32 \
  --d_model 256 \
  --d_ff 512 \
  --n_heads 8 \
  --seq_len 96 \
  --loss_type focal \
  --focal_alpha 1.0 \
  --focal_gamma 2.0 \
  --early_stopping_metric auc_pr \
  --des 'FocalLoss' \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 15

# Experiment 2: Weighted Cross-Entropy
python -u run_imbalanced.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./data/ \
  --data_path $data_path \
  --model_id imbalanced_weighted_ce \
  --model $model_name \
  --data custom \
  --label_col label \
  --e_layers 3 \
  --batch_size 32 \
  --d_model 256 \
  --d_ff 512 \
  --n_heads 8 \
  --seq_len 96 \
  --loss_type weighted_ce \
  --pos_weight_scale 2.0 \
  --early_stopping_metric auc_pr \
  --des 'WeightedCE' \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 15

# Experiment 3: Focal Loss with higher gamma for extreme imbalance
python -u run_imbalanced.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./data/ \
  --data_path $data_path \
  --model_id imbalanced_focal_high_gamma \
  --model $model_name \
  --data custom \
  --label_col label \
  --e_layers 3 \
  --batch_size 64 \
  --d_model 256 \
  --d_ff 512 \
  --n_heads 8 \
  --seq_len 96 \
  --loss_type focal \
  --focal_alpha 2.0 \
  --focal_gamma 3.0 \
  --early_stopping_metric auc_pr \
  --des 'FocalHighGamma' \
  --itr 1 \
  --learning_rate 0.0005 \
  --train_epochs 100 \
  --patience 15

# Experiment 4: Larger model with focal loss
python -u run_imbalanced.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./data/ \
  --data_path $data_path \
  --model_id imbalanced_large_model \
  --model $model_name \
  --data custom \
  --label_col label \
  --e_layers 4 \
  --batch_size 16 \
  --d_model 512 \
  --d_ff 1024 \
  --n_heads 16 \
  --seq_len 96 \
  --loss_type focal \
  --focal_alpha 1.5 \
  --focal_gamma 2.5 \
  --early_stopping_metric auc_pr \
  --des 'LargeModel' \
  --itr 1 \
  --learning_rate 0.0005 \
  --train_epochs 100 \
  --patience 20

echo "All experiments completed!"
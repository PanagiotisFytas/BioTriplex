# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass

from typing import Tuple, Union

@dataclass
class train_config:
    model_name: str="PATH/to/Model"
    tokenizer_name: str=None
    enable_fsdp: bool=False # shards model parameters, optimizer states and gradients across DDP ranks
    low_cpu_fsdp: bool=False # saves cpu memory by loading pretrained model on rank0 only
    run_validation: bool=True
    batch_size_training: int=4
    batching_strategy: str="packing" #alternative: padding
    context_length: int=4096
    gradient_accumulation_steps: int=1
    gradient_clipping: bool = False
    gradient_clipping_threshold: float = 1.0
    num_epochs: int=3
    max_train_step: int=0
    max_eval_step: int=0
    num_workers_dataloader: int=1
    lr: float=1e-4
    weight_decay: float=0.0
    gamma: float= 0.85 # multiplicatively decay the learning rate by gamma after each epoch
    seed: int=42
    use_fp16: bool=False
    mixed_precision: bool=True
    val_batch_size: int=1
    dataset = "samsum_dataset"
    peft_method: str = "lora" # None, llama_adapter (Caution: llama_adapter is currently not supported with FSDP)
    use_peft: bool=False # use parameter efficient fine tuning
    from_peft_checkpoint: str="" # if not empty and use_peft=True, will load the peft checkpoint and resume the fine-tuning on that checkpoint
    output_dir: str = "PATH/to/save/PEFT/model"
    freeze_layers: bool = False
    num_freeze_layers: int = 1
    quantization: str = None
    one_gpu: bool = False
    save_model: bool = True
    dist_checkpoint_root_folder: str="PATH/to/save/FSDP/model" # will be used if using FSDP
    dist_checkpoint_folder: str="fine-tuned" # will be used if using FSDP
    save_optimizer: bool=False # will be used if using FSDP
    use_fast_kernels: bool = False # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    use_wandb: bool = False # Enable wandb for experient tracking
    save_metrics: bool = False # saves training metrics to a json file for later plotting
    flop_counter: bool = False # Enable flop counter to measure model throughput, can not be used with pytorch profiler at the same time.
    flop_counter_start: int = 3 # The step to start profiling, default is 3, which means after 3 steps of warmup stage, the profiler will start to count flops.
    use_profiler: bool = False # Enable pytorch profiler, can not be used with flop counter at the same time.
    profiler_dir: str = "PATH/to/save/profiler/results" # will be used if using profiler
    use_entity_tokens_as_targets: bool = False # Use entity tokens as targets for the model training loss
    entity_special_tokens: bool = False # Use entity special tokens for the model training loss. Otherwise, use pretrained non-special tokens.
    upweight_minority_class: bool = False # Upweight the minority class in the dataset
    single_machine_two_gpus: bool = False # Use two GPUs on a single machine
    bidirectional_attention_in_entity_tokens: bool = False # Use bidirectional attention in entity tokens
    shift_entity_tokens: bool = False # Shift entity tokens
    weighted_sampling: bool = False # Use weighted sampling
    upsampling_positive_factor: float = 1.0 # Upsampling positive factor
    return_neg_relations: bool = False # Return negative relations
    general_relations: bool = False # Use general relations in the dataset
    background: bool = False # Use background in the dataset
    group_relations: bool = True # Group relations for qa dataset
    use_embeddings: bool = False # Enable using embeddings as additional modality
    # Dimension of the input embeddings int or tuple of int
    embedding_dim: int = 768 # Default is 768 for BERT-like models
    pooling_type: str = "mean" # Pooling type for the input embeddings. Options: "mean", "attention", "sequence"
    train_sample_pct: float = 1.0 # Percentage of training samples to use
    train_sample_seed: int = 42
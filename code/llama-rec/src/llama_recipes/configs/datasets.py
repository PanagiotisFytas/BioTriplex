# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass


@dataclass
class samsum_dataset:
    dataset: str =  "samsum_dataset"
    train_split: str = "train"
    test_split: str = "validation"


@dataclass
class grammar_dataset:
    dataset: str = "grammar_dataset"
    train_split: str = "src/llama_recipes/datasets/grammar_dataset/gtrain_10k.csv"
    test_split: str = "src/llama_recipes/datasets/grammar_dataset/grammar_validation.csv"


@dataclass
class alpaca_dataset:
    dataset: str = "alpaca_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "src/llama_recipes/datasets/alpaca_data.json"

@dataclass
class custom_dataset:
    dataset: str = "custom_dataset"
    file: str = "recipes/quickstart/finetuning/datasets/custom_dataset.py"
    train_split: str = "train"
    test_split: str = "validation"
    data_path: str = ""

@dataclass
class llamaguard_toxicchat_dataset:
    dataset: str = "llamaguard_toxicchat_dataset"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class biotriplex_ner_dataset:
    dataset: str = "biotriplex_ner_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "../biotriplex/data/"
    use_entity_tokens_as_targets: bool = False
    entity_special_tokens: bool = False
    upweight_minority_class: bool = False
    bidirectional_attention_in_entity_tokens: bool = False
    shift_entity_tokens: bool = False
    num_of_shots: int = 4

@dataclass
class biotriplex_nerlong_dataset:
    dataset: str = "biotriplex_nerlong_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "../biotriplex/data/"
    use_entity_tokens_as_targets: bool = False
    entity_special_tokens: bool = False
    upweight_minority_class: bool = False
    bidirectional_attention_in_entity_tokens: bool = False
    shift_entity_tokens: bool = False
    num_of_shots: int = 4


@dataclass
class biored_qa_dataset:
    dataset: str = "biored_qa_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "../biotriplex/data/biored/"
    use_entity_tokens_as_targets: bool = False
    entity_special_tokens: bool = False
    upweight_minority_class: bool = False
    bidirectional_attention_in_entity_tokens: bool = False
    shift_entity_tokens: bool = False
    num_of_shots: int = 4

class biored_qakshot_dataset(biored_qa_dataset):
    pass

@dataclass()
class biotriplex_qa_dataset:
    dataset: str = "biotriplex_qa_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "../biotriplex/data/"
    use_entity_tokens_as_targets: bool = False
    entity_special_tokens: bool = False
    upweight_minority_class: bool = False
    bidirectional_attention_in_entity_tokens: bool = False
    shift_entity_tokens: bool = False
    return_neg_relations: bool = False
    general_relations: bool = False
    num_of_shots: int = 0
    group_relations: bool = True
    train_sample_pct: float = 1.0
    train_sample_seed: int = 42
    train_sample_stratify: bool = False
    train_sample_min_per_label: int = 0

class biotriplex_qakshot_dataset(biotriplex_qa_dataset):
    pass

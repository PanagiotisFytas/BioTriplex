# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
from functools import partial

from llama_recipes.datasets.grammar_dataset.grammar_dataset import get_dataset as get_grammar_dataset
from llama_recipes.datasets.alpaca_dataset import InstructionDataset as get_alpaca_dataset
from llama_recipes.datasets.custom_dataset import get_custom_dataset,get_data_collator
from llama_recipes.datasets.samsum_dataset import get_preprocessed_samsum as get_samsum_dataset
from llama_recipes.datasets.toxicchat_dataset import get_llamaguard_toxicchat_dataset as get_llamaguard_toxicchat_dataset
from llama_recipes.datasets.biotriplex_dataset import BioTriplexDataset as get_biotriplex_dataset
from llama_recipes.datasets.biotriplex_ner_dataset import BioTriplexNERDataset as get_biotriplex_ner_dataset
from llama_recipes.datasets.biotriplex_ner0shot_dataset import BioTriplexNERDataset as get_biotriplex_ner0shot_dataset
from llama_recipes.datasets.biotriplex_nerkshot_dataset import BioTriplexNERDataset as get_biotriplex_nerkshot_dataset
from llama_recipes.datasets.biotriplex_nerlong_dataset import BioTriplexNERLongDataset as get_biotriplex_nerlong_dataset
from llama_recipes.datasets.biotriplex_nerlong0shot_dataset import BioTriplexNERLongDataset as get_biotriplex_nerlong0shot_dataset
from llama_recipes.datasets.biotriplex_nerlongkshot_dataset import BioTriplexNERLongDataset as get_biotriplex_nerlongkshot_dataset
from llama_recipes.datasets.biotriplex_rel_dataset import BioTriplexRelDataset as get_biotriplex_rel_dataset
from llama_recipes.datasets.biored_dataset import BioRedDataset as get_biored_dataset
from llama_recipes.datasets.biored_qa_dataset import BioRedQADataset as get_biored_qa_dataset
from llama_recipes.datasets.biored_qakshot_dataset import BioRedQADataset as get_biored_qakshot_dataset
# from llama_recipes.datasets.biored_qa0shot_dataset import BioRedQADataset as get_biored_qa0shot_dataset
from llama_recipes.datasets.biotriplex_qa_dataset import BioTriplexQADataset as get_biotriplex_qa_dataset
from llama_recipes.datasets.biotriplex_qa0shot_dataset import BioTriplexQADataset as get_biotriplex_qa0shot_dataset
from llama_recipes.datasets.biotriplex_qakshot_dataset import BioTriplexQADataset as get_biotriplex_qakshot_dataset
from llama_recipes.datasets.mimic_impression import MimicImpressionDataset as get_mimic_impression_dataset


DATASET_PREPROC = {
    "alpaca_dataset": partial(get_alpaca_dataset),
    "grammar_dataset": get_grammar_dataset,
    "samsum_dataset": get_samsum_dataset,
    "custom_dataset": get_custom_dataset,
    "llamaguard_toxicchat_dataset": get_llamaguard_toxicchat_dataset,
    "biotriplex_dataset": get_biotriplex_dataset,
    "biotriplex_ner_dataset": get_biotriplex_ner_dataset,
    "biotriplex_ner0shot_dataset": get_biotriplex_ner0shot_dataset,
    "biotriplex_nerkshot_dataset": get_biotriplex_nerkshot_dataset,
    "biotriplex_nerlong_dataset": get_biotriplex_nerlong_dataset,
    "biotriplex_nerlong0shot_dataset": get_biotriplex_nerlong0shot_dataset,
    "biotriplex_nerlongkshot_dataset": get_biotriplex_nerlongkshot_dataset,
    "biotriplex_rel_dataset": get_biotriplex_rel_dataset,
    "biored_dataset": get_biored_dataset,
    "biored_qa_dataset": get_biored_qa_dataset,
    "biored_qakshot_dataset": get_biored_qakshot_dataset,
    "biotriplex_qa_dataset": get_biotriplex_qa_dataset,
    # "biored_qa0shot_dataset": get_biored_qa0shot_dataset,
    "biotriplex_qa0shot_dataset": get_biotriplex_qa0shot_dataset,
    "biotriplex_qakshot_dataset": get_biotriplex_qakshot_dataset,
    "mimic_impression_dataset": get_mimic_impression_dataset
}
DATALOADER_COLLATE_FUNC = {
    "custom_dataset": get_data_collator
}

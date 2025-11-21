# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import sys
import time

import fire

import torch

from accelerate.utils import is_xpu_available
from llama_recipes.inference.model_utils import load_model, load_peft_model

from llama_recipes.inference.safety_utils import AgentType, get_safety_checker
from requests.packages import target
from transformers import AutoTokenizer

import tqdm
from guidance import models, gen, zero_or_more, select, one_or_more
import guidance
import re

# @guidance(stateless=True, )
# def constraint_prompt(llm, prompt):
#     return llm + prompt + "\n### Response:\n" + \
#            "[" + one_or_more(
#         select([
#             f"""{{"span": "{gen(stop='",', max_tokens=50)}", "entity_type": "GENE"}}, """,
#             f"""{{"span": "{gen(stop='",', max_tokens=50)}", "entity_type": "DISEASE"}}, """,
#             f"""{{"span": "{gen(stop='",', max_tokens=50)}", "entity_type": "RELATION"}}, """,])) + \
#            f"""{{"span": "{gen(stop='",', max_tokens=50)}", "entity_type": "{select(['GENE', 'DISEASE', 'RELATION'])}"}}]"""
#
#     # select([
#         #     "[" + zero_or_more(
#         #         f"""{{'span': "{gen(stop='"', max_tokens=50)}", "entity_type": "{select(['GENE', 'DISEASE', 'RELATION'])}"}}, """) + \
#         #     f"""{{'span': "{gen(stop='"', max_tokens=50)}", "entity_type": "{select(['GENE', 'DISEASE', 'RELATION'])}"}}]""",
#         #     "[]"
#         # ])

@guidance(stateless=True)
def ner_instruction(llm, prompt):
    llm += prompt + "\n### Response:\n"
    return llm

@guidance(stateless=True)
def constraint_prompt(llm, prompt):
    input_text = \
        prompt.split("### Input:\n")[1].split("\n\n" + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>")[0]
    words = [input_text[m.start():m.end()] for m in re.finditer(r'\w+|[^\w\s]', input_text)]
    ret = ''
    word_no = 0
    for word in words:
        # ret += word + ": " + select(["GENE", "DISEASE", "RELATION", "NULL"], name=f"{word_no}") + "\n"
        ret += word + ":" + gen(stop='\n', max_tokens=10) + "\n"
        word_no += 1
    return llm + ret


def main(
    model_name,
    peft_model: str = None,
    quantization: str = None, # Options: 4bit, 8bit
    max_new_tokens=100,  # The maximum numbers of tokens to generate
    prompt_file: str = None,
    seed: int = 42,  # seed value for reproducibility
    do_sample: bool = True,  # Whether or not to use sampling ; use greedy decoding otherwise.
    min_length: int = None,  # The minimum length of the sequence to be generated, input prompt + min_new_tokens
    use_cache: bool = True,  # [optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float = 1.0,  # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float = 1.0,  # [optional] The value used to modulate the next token probabilities.
    top_k: int = 50,  # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float = 1.0,  # The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int = 1,  # [optional] Exponential penalty to the length that is used with beam-based generation.
    enable_azure_content_safety: bool = False,  # Enable safety check with Azure content safety api
    enable_sensitive_topics: bool = False,  # Enable check for sensitive topics using AuditNLG APIs
    enable_salesforce_content_safety: bool = True,  # Enable safety check with Salesforce safety flan t5
    enable_llamaguard_content_safety: bool = False,
    max_padding_length: int = None,  # the max padding length to be used with tokenizer padding the prompts.
    use_fast_kernels: bool = False,  # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    share_gradio: bool = False,  # Enable endpoint creation for gradio.live
    full_dataset: bool = False,  # Enable full dataset inference
    dataset_mode: str = "val",  # The dataset mode to be used with full dataset inference
    ner_dataset: bool = False,  # Enable NER dataset
    rel_dataset: bool = False,  # Enable relation dataset
    qa_dataset: bool = False,  # Enable QA dataset
    biored_dataset: bool = False,  # Enable BioRED dataset
    biored_qa_dataset: bool = False,  # Enable BioRED QA dataset
    mimic_impression_dataset: bool = False,  # Enable MIMIC Impression dataset
    background: bool = False,  # Use background in the MIMIC Impression dataset
    prior_report: bool = False,  # Use prior report in the MIMIC Impression dataset
    use_entity_tokens_as_targets: bool = False,  # Use entity tokens as targets for the model training loss
    entity_special_tokens: bool = False,  # Use entity special tokens for the model training loss. Otherwise, use pretrained non-special tokens.
    bidirectional_attention_in_entity_tokens: bool = False,  # Use bidirectional attention in entity tokens,
    shift_entity_tokens: bool = False,  # Shift entity tokens
    general_relations: bool = False,  # Use general relations in the dataset
    return_neg_relations: bool = False,  # Return negative relations
    prefix: str = "",  # Prefix of output file
    max_entities: int = 100,  # Maximum number of entities in the dataset
    max_tokens_per_entity: int = 50,  # Maximum number of tokens per entity in the dataset
     **kwargs,
):
    # Set the seeds for reproducibility
    if is_xpu_available():
        torch.xpu.manual_seed(seed)
    else:
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    if use_entity_tokens_as_targets and entity_special_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": ["<|gene token|>",
                                                                    "<|disease token|>",
                                                                    "<|relation token|>",
                                                                    "<|no entity token|>"]})

    model = load_model(model_name, quantization, use_fast_kernels,
                       bidirectional=bidirectional_attention_in_entity_tokens, **kwargs)

    # If there is a mismatch between tokenizer vocab size and embedding matrix,
    # throw a warning and then expand the embedding matrix
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        print("WARNING: Resizing the embedding matrix to match the tokenizer vocab size.")
        model.resize_token_embeddings(len(tokenizer))

    if peft_model:
        model = load_peft_model(model, peft_model, bidirectional=bidirectional_attention_in_entity_tokens)

    model.eval()
    print("Loading Guidance Model")
    print("Guidance Model Loaded")
    llm = models.Transformers(model, tokenizer)

    def inference(
        user_prompt,
        temperature,
        top_p,
        top_k,
        max_new_tokens,
        length_penalty,
        **kwargs,
    ):
        llm.reset()
        start = time.perf_counter()
        outputs = llm + ner_instruction(user_prompt) + constraint_prompt(user_prompt)
        e2e_inference_time = (time.perf_counter() - start) * 1000
        print(f"the inference time is {e2e_inference_time} ms")
        print(outputs)
        print(str(outputs))
        # output_dict = []
        # for n in range(outputs["entities"]):
        #     output_dict.append(
        #         {
        #             "span": outputs[f"span{n}"],
        #             "entity_type": outputs[f"entity_type{n}"]
        #         }
        #     )
        # print(output_dict)
        # outputs_json_string = json.dumps(output_dict)
        return outputs

    if prompt_file is not None:
        raise NotImplementedError("Prompt file not implemented with guidance")
    elif full_dataset:
        outputs = {}
        from llama_recipes.configs.datasets import biotriplex_dataset, biotriplex_ner_dataset, biotriplex_rel_dataset
        import json
        if ner_dataset:
            assert not rel_dataset, "Cannot have both NER and relation datasets"
            from llama_recipes.datasets.biotriplex_ner_dataset import BioTriplexNERDataset
            from llama_recipes.configs.datasets import biotriplex_ner_dataset
            biotriplex_ner_dataset.use_entity_tokens_as_targets = use_entity_tokens_as_targets
            biotriplex_ner_dataset.entity_special_tokens = entity_special_tokens
            biotriplex_ner_dataset.bidirectional_attention_in_entity_tokens = bidirectional_attention_in_entity_tokens
            biotriplex_ner_dataset.shift_entity_tokens = shift_entity_tokens
            dataset = BioTriplexNERDataset(biotriplex_ner_dataset, tokenizer, dataset_mode, max_words=None)
        elif rel_dataset:
            from llama_recipes.datasets.biotriplex_rel_dataset import BioTriplexRelDataset
            from llama_recipes.configs.datasets import biotriplex_rel_dataset
            biotriplex_rel_dataset.use_entity_tokens_as_targets = use_entity_tokens_as_targets
            biotriplex_rel_dataset.entity_special_tokens = entity_special_tokens
            biotriplex_rel_dataset.bidirectional_attention_in_entity_tokens = bidirectional_attention_in_entity_tokens
            biotriplex_rel_dataset.shift_entity_tokens = shift_entity_tokens
            dataset = BioTriplexRelDataset(biotriplex_rel_dataset, tokenizer, dataset_mode, max_words=None)
        elif qa_dataset:
            from llama_recipes.datasets.biotriplex_qa_dataset import BioTriplexQADataset
            from llama_recipes.configs.datasets import biotriplex_qa_dataset
            biotriplex_qa_dataset.use_entity_tokens_as_targets = use_entity_tokens_as_targets
            biotriplex_qa_dataset.entity_special_tokens = entity_special_tokens
            biotriplex_qa_dataset.bidirectional_attention_in_entity_tokens = bidirectional_attention_in_entity_tokens
            biotriplex_qa_dataset.shift_entity_tokens = shift_entity_tokens
            biotriplex_qa_dataset.general_relations = general_relations
            biotriplex_qa_dataset.return_neg_relations = return_neg_relations
            dataset = BioTriplexQADataset(biotriplex_qa_dataset, tokenizer, dataset_mode, max_words=None)
        elif biored_dataset:
            from llama_recipes.datasets.biored_dataset import BioRedDataset
            from llama_recipes.configs.datasets import biored_dataset
            biored_dataset.use_entity_tokens_as_targets = use_entity_tokens_as_targets
            biored_dataset.entity_special_tokens = entity_special_tokens
            biored_dataset.bidirectional_attention_in_entity_tokens = bidirectional_attention_in_entity_tokens
            biored_dataset.shift_entity_tokens = shift_entity_tokens
            dataset = BioRedDataset(biored_dataset, tokenizer, dataset_mode, max_words=None)
        elif biored_qa_dataset:
            from llama_recipes.datasets.biored_qa_dataset import BioRedQADataset
            from llama_recipes.configs.datasets import biored_qa_dataset
            biored_qa_dataset.use_entity_tokens_as_targets = use_entity_tokens_as_targets
            biored_qa_dataset.entity_special_tokens = entity_special_tokens
            biored_qa_dataset.bidirectional_attention_in_entity_tokens = bidirectional_attention_in_entity_tokens
            biored_qa_dataset.shift_entity_tokens = shift_entity_tokens
            dataset = BioRedQADataset(biored_qa_dataset, tokenizer, dataset_mode, max_words=None)
        elif mimic_impression_dataset:
            from llama_recipes.datasets.mimic_impression import MimicImpressionDataset
            from llama_recipes.configs.datasets import mimic_impression_dataset
            mimic_impression_dataset.background = background
            mimic_impression_dataset.prior_report = prior_report
            dataset = MimicImpressionDataset(mimic_impression_dataset, tokenizer, dataset_mode, max_words=None)
            gold_output = dataset.get_gold_outputs()
        else:
            from llama_recipes.datasets.biotriplex_dataset import BioTriplexDataset
            from llama_recipes.configs.datasets import biotriplex_dataset
            biotriplex_dataset.use_entity_tokens_as_targets = use_entity_tokens_as_targets
            biotriplex_dataset.entity_special_tokens = entity_special_tokens
            biotriplex_dataset.bidirectional_attention_in_entity_tokens = bidirectional_attention_in_entity_tokens
            biotriplex_dataset.shift_entity_tokens = shift_entity_tokens
            dataset = BioTriplexDataset(biotriplex_dataset, tokenizer, dataset_mode, max_words=None)
        for doc_key, prompt in tqdm.tqdm(dataset.get_all_input_prompts(
                bidirectional=bidirectional_attention_in_entity_tokens
        ).items()):
            if bidirectional_attention_in_entity_tokens:
                raise NotImplementedError("Bidirectional attention not implemented with guidance")

            output = inference(prompt, temperature, top_p, top_k, max_new_tokens, length_penalty)
            outputs[doc_key] = output
        # Save the outputs to a file
        if rel_dataset:
            dataset_str = "biotriplex_rel"
        elif ner_dataset:
            dataset_str = "biotriplex_ner"
        elif biored_dataset:
            dataset_str = "biored"
        elif biored_qa_dataset:
            dataset_str = "biored_qa"
        elif qa_dataset:
            dataset_str = "biotriplex_qa"
        elif mimic_impression_dataset:
            dataset_str = "mimic_impression"
        else:
            dataset_str = "biotriplex"
        target_filename = f"{prefix}{dataset_str}_model_{model_name.split('/')[-1]}_" +\
                            f"peft_{peft_model.split('/')[-1] if peft_model else 'no_peft'}_" +\
                            f"quantization_{quantization if quantization else 'no_quantization'}_" +\
                            f"mode_{dataset_mode}_guidance.json"
                            # f"use_entity_tokens_as_targets_{use_entity_tokens_as_targets}_" +\
                            # f"entity_special_tokens_{entity_special_tokens}_" +\
                            # f"bidirectional_attention_in_entity_tokens_{bidirectional_attention_in_entity_tokens}_" +\
        with open(target_filename, "w") as f:
            json.dump(outputs, f)
        if mimic_impression_dataset:
            raise NotImplementedError("MIMIC Impression dataset not implemented with guidance")
    else:
        try:
            import gradio as gr
        except ImportError:
            raise ImportError("This part of the recipe requires gradio. Please run `pip install gradio`")

        gr.Interface(
            fn=inference,
            inputs=[
                gr.components.Textbox(
                    lines=9,
                    label="User Prompt",
                    placeholder="none",
                ),
                gr.components.Slider(
                    minimum=0, maximum=1, value=1.0, label="Temperature"
                ),
                gr.components.Slider(minimum=0, maximum=1, value=1.0, label="Top p"),
                gr.components.Slider(
                    minimum=0, maximum=100, step=1, value=50, label="Top k"
                ),
                gr.components.Slider(
                    minimum=1, maximum=2000, step=1, value=200, label="Max tokens"
                ),
                gr.components.Slider(
                    minimum=-10, maximum=10, step=1, value=1, label="Length penalty"
                ),
            ],
            outputs=[
                gr.components.Textbox(
                    lines=5,
                    label="Output",
                )
            ],
            title="Meta Llama3 Playground",
            description="https://github.com/meta-llama/llama-recipes",
        ).queue().launch(server_name="0.0.0.0", share=share_gradio)


if __name__ == "__main__":
    fire.Fire(main)

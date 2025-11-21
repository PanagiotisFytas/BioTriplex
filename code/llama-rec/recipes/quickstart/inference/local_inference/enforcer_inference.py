# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import sys
import time
from typing import Optional

import fire

import torch

from accelerate.utils import is_xpu_available
from llama_recipes.inference.model_utils import load_model, load_peft_model

from llama_recipes.inference.safety_utils import AgentType, get_safety_checker
from requests.packages import target
from transformers import AutoTokenizer
from lmformatenforcer import RegexParser, CharacterLevelParser, StringParser, JsonSchemaParser
from lmformatenforcer.integrations.transformers import generate_enforced, build_token_enforcer_tokenizer_data
from transformers.generation.logits_process import LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper
from types import MethodType
import tqdm
import re


def _get_logits_warper(
        self,
        generation_config,
) -> LogitsProcessorList:
    """
    This class returns a [`TFLogitsProcessorList`] list object that contains all relevant [`TFLogitsWarper`]
    instances used for multinomial sampling.
    """

    # instantiate warpers list
    warpers = LogitsProcessorList()

    # In beam methods, we need to keep at least one non-eos token to explore continuations that might have a
    # better score (i.e. keep len(generation_config.eos_token_id) + 1)
    if generation_config.num_beams > 1:
        if isinstance(generation_config.eos_token_id, list):
            min_tokens_to_keep = len(generation_config.eos_token_id) + 1
        else:
            min_tokens_to_keep = 2
    else:
        min_tokens_to_keep = 1

    if generation_config.temperature is not None and generation_config.temperature != 1.0:
        warpers.append(TemperatureLogitsWarper(generation_config.temperature))
    if generation_config.top_k is not None and generation_config.top_k != 0:
        warpers.append(TopKLogitsWarper(top_k=generation_config.top_k, min_tokens_to_keep=min_tokens_to_keep))
    if generation_config.top_p is not None and generation_config.top_p < 1.0:
        warpers.append(TopPLogitsWarper(top_p=generation_config.top_p, min_tokens_to_keep=min_tokens_to_keep))
    return warpers




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
    nerlong_dataset: bool = False,  # Enable NER long dataset
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
    model._get_logits_warper = MethodType(_get_logits_warper, model)
    tokenizer_data = build_token_enforcer_tokenizer_data(tokenizer)


    def inference(
        user_prompt,
        required_constraint,
        temperature,
        top_p,
        top_k,
        max_new_tokens,
        length_penalty,
        constraint_type="regex",
        **kwargs,
    ):
        safety_checker = get_safety_checker(
            enable_azure_content_safety,
            enable_sensitive_topics,
            enable_salesforce_content_safety,
            enable_llamaguard_content_safety,
        )

        # Safety check of the user prompt
        safety_results = [check(user_prompt) for check in safety_checker]
        are_safe = all([r[1] for r in safety_results])
        if are_safe:
            print("User prompt deemed safe.")
            print(f"User prompt:\n{user_prompt}")
        else:
            print("User prompt deemed unsafe.")
            for method, is_safe, report in safety_results:
                if not is_safe:
                    print(method)
                    print(report)
            print("Skipping the inference as the prompt is not safe.")
            return  # Exit the program with an error status

        user_prompt = [user_prompt]
        if is_xpu_available():
            device = "xpu"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        inputs = tokenizer(user_prompt, return_tensors='pt', add_special_tokens=False, return_token_type_ids=False,
                           padding=False).to(device)
        generate_kwargs = dict(
            inputs,
            # streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            num_beams=1,
            output_scores=False,
            return_dict_in_generate=True
        )
        parser: Optional[CharacterLevelParser] = None
        if constraint_type == "regex":
            parser = RegexParser(required_constraint)
        elif constraint_type == "string":
            parser = StringParser(required_constraint)
        elif constraint_type == "json-schema":
            parser = JsonSchemaParser(required_constraint)
        elif constraint_type == "json-output":
            parser = JsonSchemaParser(None)
        assert constraint_type == "regex", "Not implemented yet"
        start = time.perf_counter()
        with torch.no_grad():
            # outputs = model.generate(
            #     **batch,
            #     max_new_tokens=max_new_tokens,
            #     do_sample=do_sample,
            #     top_p=top_p,
            #     temperature=temperature,
            #     min_length=min_length,
            #     use_cache=use_cache,
            #     top_k=top_k,
            #     repetition_penalty=repetition_penalty,
            #     length_penalty=length_penalty,
            #     **kwargs,
            # )
            outputs = generate_enforced(model, tokenizer_data, parser, **generate_kwargs)['sequences']
        e2e_inference_time = (time.perf_counter() - start) * 1000
        print(f"the inference time is {e2e_inference_time} ms")
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Safety check of the model output
        safety_results = [
            check(output_text, agent_type=AgentType.AGENT, user_prompt=user_prompt)
            for check in safety_checker
        ]
        are_safe = all([r[1] for r in safety_results])
        if are_safe:
            print("User input and model output deemed safe.")
            print(f"Model output:\n{output_text}")
            return output_text
        else:
            print("Model output deemed unsafe.")
            for method, is_safe, report in safety_results:
                if not is_safe:
                    print(method)
                    print(report)
            return None

    if prompt_file is not None:
        assert os.path.exists(
            prompt_file
        ), f"Provided Prompt file does not exist {prompt_file}"
        with open(prompt_file, "r") as f:
            user_prompt = "\n".join(f.readlines())
        inference(user_prompt, temperature, top_p, top_k, max_new_tokens, length_penalty)
    # elif not sys.stdin.isatty():
    #     user_prompt = "\n".join(sys.stdin.readlines())
    #     inference(user_prompt, temperature, top_p, top_k, max_new_tokens, length_penalty)
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
        elif nerlong_dataset:
            assert not rel_dataset, "Cannot have both NER and relation datasets"
            from llama_recipes.datasets.biotriplex_nerlong_dataset import BioTriplexNERLongDataset
            from llama_recipes.configs.datasets import biotriplex_nerlong_dataset
            biotriplex_nerlong_dataset.use_entity_tokens_as_targets = use_entity_tokens_as_targets
            biotriplex_nerlong_dataset.entity_special_tokens = entity_special_tokens
            biotriplex_nerlong_dataset.bidirectional_attention_in_entity_tokens = bidirectional_attention_in_entity_tokens
            biotriplex_nerlong_dataset.shift_entity_tokens = shift_entity_tokens
            dataset = BioTriplexNERLongDataset(biotriplex_nerlong_dataset, tokenizer, dataset_mode, max_words=None)
            constraint_type = "regex"
            constraint = r"(\s*\w+:\s*(NULL|GENE|DISEASE|RELATION)(,?\s*)?\n?)+"
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
                bidirectional_region_start = prompt["bidirectional_region_start"]
                bidirectional_region_end = prompt["bidirectional_region_end"]
                prompt = prompt["prompt"]
                # add the bidirectional region start and end to kwargs
                assert bidirectional_region_start is not None and bidirectional_region_end is not None, "Bidirectional region start and end must be provided"
                model.set_bidirectional_region(bidirectional_region_start, bidirectional_region_end)
            if not nerlong_dataset:
                raise NotImplementedError("Only NER long dataset is supported for full dataset inference")
            input_text = \
                prompt.split("### Input:\n")[1].split(
                    "\n\n" + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>")[0]
            words = [input_text[m.start():m.end()] for m in re.finditer(r'\w+|[^\w\s]', input_text)]
            constraint = ""
            idx = 0
            for word in words:
                last_word = idx == len(words) - 1
                idx += 1
                if not last_word:
                    constraint += f'(\s*{word}:\s*(NULL|GENE|DISEASE|RELATION)(,?\s*)?\n)'
                else:
                    constraint += f'(\s*{word}:\s*(NULL|GENE|DISEASE|RELATION)(,?\s*)?)'
            output = inference(prompt, constraint, temperature, top_p, top_k, max_new_tokens, length_penalty, constraint_type=constraint_type)
            outputs[doc_key] = output
        # Save the outputs to a file
        if rel_dataset:
            dataset_str = "biotriplex_rel"
        elif ner_dataset:
            dataset_str = "biotriplex_ner"
        elif nerlong_dataset:
            dataset_str = "biotriplex_nerlong"
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
                            f"mode_{dataset_mode}_enforcer.json"
                            # f"use_entity_tokens_as_targets_{use_entity_tokens_as_targets}_" +\
                            # f"entity_special_tokens_{entity_special_tokens}_" +\
                            # f"bidirectional_attention_in_entity_tokens_{bidirectional_attention_in_entity_tokens}_" +\
        with open(target_filename, "w") as f:
            json.dump(outputs, f)
        if mimic_impression_dataset:
            gold_filename = f"{prefix}mimic_impression_peft_{peft_model.split('/')[-1] if peft_model else 'no_peft'}_" +\
                            f"mode_{dataset_mode}_gold_outputs.json"
            with open(gold_filename, "w") as f:
                json.dump(gold_output, f)
            # save them also as csv.
            # Ground Truth and Predicted reports must be arranged in the same order in a column named "report" in two CSV files.
            # The CSVs should also contain a corresponding "study_id" column (with the doc_key),
            # that contains unique identifies for the reports.
            # The CSVs should be named as the json filenames with the json extension replaced with csv.
            import pandas as pd
            gold_df = pd.DataFrame(gold_output.items(), columns=["study_id", "report"])
            # sort the gold_df by study_id
            gold_df = gold_df.sort_values(by="study_id")
            gold_df = MimicImpressionDataset.clean_text(gold_df, ["report"])
            gold_df.to_csv(gold_filename.replace(".json", ".csv"), index=False)
            output_df = pd.DataFrame(outputs.items(), columns=["study_id", "report"])
            # sort the output_df by study_id
            output_df = output_df.sort_values(by="study_id")
            output_df = MimicImpressionDataset.clean_text(output_df, ["report"])
            output_df.to_csv(target_filename.replace(".json", ".csv"), index=False)
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

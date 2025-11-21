# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
import json
import os
import time
import fire
import torch
from accelerate.utils import is_xpu_available
from llama_recipes.inference.model_utils import load_model, load_peft_model
from llama_recipes.inference.safety_utils import AgentType, get_safety_checker
from requests.packages import target
from transformers import AutoTokenizer

import tqdm


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
    ner_kshot_dataset: bool = False,  # Enable NER k-shot dataset
    nerlong_dataset: bool = False,  # Enable NER long dataset
    nerlong_kshot_dataset: bool = False,  # Enable NER long few-shot dataset
    qa_kshot_dataset: bool = False,  # Enable QA k-shot dataset
    biored_qakshot_dataset: bool = False,  # Enable BioRED QA k-shot dataset
    background: bool = False,  # Use background in the MIMIC Impression dataset
    prior_report: bool = False,  # Use prior report in the MIMIC Impression dataset
    use_entity_tokens_as_targets: bool = False,  # Use entity tokens as targets for the model training loss
    entity_special_tokens: bool = False,  # Use entity special tokens for the model training loss. Otherwise, use pretrained non-special tokens.
    bidirectional_attention_in_entity_tokens: bool = False,  # Use bidirectional attention in entity tokens,
    shift_entity_tokens: bool = False,  # Shift entity tokens
    general_relations: bool = False,  # Use general relations in the dataset
    return_neg_relations: bool = False,  # Return negative relations
    prefix: str = "",  # Prefix of output file
    num_of_shots: int = 0,  # Number of shots for few-shot learning
    group_relations: bool = True,  # Group relations for qa dataset
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


    def inference(
        user_prompt,
        temperature,
        top_p,
        top_k,
        max_new_tokens,
        length_penalty,
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

        batch = tokenizer(
            user_prompt,
            truncation=True,
            max_length=max_padding_length,
            return_tensors="pt",
        )
        if is_xpu_available():
            batch = {k: v.to("xpu") for k, v in batch.items()}
        else:
            batch = {k: v.to("cuda") for k, v in batch.items()}

        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                min_length=min_length,
                use_cache=use_cache,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                **kwargs,
            )
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
        if ner_dataset:
            assert not rel_dataset, "Cannot have both NER and relation datasets"
            from llama_recipes.datasets.biotriplex_ner_dataset import BioTriplexNERDataset
            from llama_recipes.configs.datasets import biotriplex_ner_dataset
            biotriplex_ner_dataset.use_entity_tokens_as_targets = use_entity_tokens_as_targets
            biotriplex_ner_dataset.entity_special_tokens = entity_special_tokens
            biotriplex_ner_dataset.bidirectional_attention_in_entity_tokens = bidirectional_attention_in_entity_tokens
            biotriplex_ner_dataset.shift_entity_tokens = shift_entity_tokens
            dataset = BioTriplexNERDataset(biotriplex_ner_dataset, tokenizer, dataset_mode, max_words=None)
        elif ner_kshot_dataset:
            assert not rel_dataset, "Cannot have both NER and relation datasets"
            from llama_recipes.datasets.biotriplex_nerkshot_dataset import BioTriplexNERDataset
            from llama_recipes.configs.datasets import biotriplex_ner_dataset
            biotriplex_ner_dataset.use_entity_tokens_as_targets = use_entity_tokens_as_targets
            biotriplex_ner_dataset.entity_special_tokens = entity_special_tokens
            biotriplex_ner_dataset.bidirectional_attention_in_entity_tokens = bidirectional_attention_in_entity_tokens
            biotriplex_ner_dataset.shift_entity_tokens = shift_entity_tokens
            biotriplex_ner_dataset.num_of_shots = num_of_shots
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
        elif nerlong_kshot_dataset:
            assert not rel_dataset, "Cannot have both NER and relation datasets"
            from llama_recipes.datasets.biotriplex_nerlongkshot_dataset import BioTriplexNERLongDataset
            from llama_recipes.configs.datasets import biotriplex_nerlong_dataset
            biotriplex_nerlong_dataset.use_entity_tokens_as_targets = use_entity_tokens_as_targets
            biotriplex_nerlong_dataset.entity_special_tokens = entity_special_tokens
            biotriplex_nerlong_dataset.bidirectional_attention_in_entity_tokens = bidirectional_attention_in_entity_tokens
            biotriplex_nerlong_dataset.shift_entity_tokens = shift_entity_tokens
            biotriplex_nerlong_dataset.num_of_shots = num_of_shots
            dataset = BioTriplexNERLongDataset(biotriplex_nerlong_dataset, tokenizer, dataset_mode, max_words=None)
        elif qa_kshot_dataset:
            from llama_recipes.datasets.biotriplex_qakshot_dataset import BioTriplexQADataset
            from llama_recipes.configs.datasets import biotriplex_qa_dataset
            biotriplex_qa_dataset.use_entity_tokens_as_targets = use_entity_tokens_as_targets
            biotriplex_qa_dataset.entity_special_tokens = entity_special_tokens
            biotriplex_qa_dataset.bidirectional_attention_in_entity_tokens = bidirectional_attention_in_entity_tokens
            biotriplex_qa_dataset.shift_entity_tokens = shift_entity_tokens
            biotriplex_qa_dataset.num_of_shots = num_of_shots
            biotriplex_qa_dataset.general_relations = general_relations
            biotriplex_qa_dataset.return_neg_relations = return_neg_relations
            biotriplex_qa_dataset.group_relations = group_relations
            dataset = BioTriplexQADataset(biotriplex_qa_dataset, tokenizer, dataset_mode, max_words=None)
        elif biored_qakshot_dataset:
            from llama_recipes.datasets.biored_qakshot_dataset import BioRedQADataset
            from llama_recipes.configs.datasets import biored_qakshot_dataset
            biored_qakshot_dataset.use_entity_tokens_as_targets = use_entity_tokens_as_targets
            biored_qakshot_dataset.entity_special_tokens = entity_special_tokens
            biored_qakshot_dataset.bidirectional_attention_in_entity_tokens = bidirectional_attention_in_entity_tokens
            biored_qakshot_dataset.shift_entity_tokens = shift_entity_tokens
            biored_qakshot_dataset.num_of_shots = num_of_shots
            dataset = BioRedQADataset(biored_qakshot_dataset, tokenizer, dataset_mode, max_words=None)
        else:
            raise NotImplementedError("No dataset selected for full dataset inference")
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
            output = inference(prompt, temperature, top_p, top_k, max_new_tokens, length_penalty)
            outputs[doc_key] = output
        # Save the outputs to a file
        if ner_dataset:
            dataset_str = "biotriplex_ner"
        elif ner_kshot_dataset:
            dataset_str = f"biotriplex_ner{num_of_shots}shot"
        elif nerlong_dataset:
            dataset_str = "biotriplex_nerlong"
        elif nerlong_kshot_dataset:
            dataset_str = f"biotriplex_nerlong{num_of_shots}shot"
        elif qa_kshot_dataset:
            dataset_str = f"biotriplex_qa{num_of_shots}shot"
        elif biored_qakshot_dataset:
            dataset_str = f"biored_qa{num_of_shots}shot"
        else:
            raise NotImplementedError("No dataset selected for full dataset inference")
        target_filename = f"{prefix}{dataset_str}_model_{model_name.split('/')[-1]}_" +\
                            f"peft_{peft_model.split('/')[-1] if peft_model else 'no_peft'}_" +\
                            f"quantization_{quantization if quantization else 'no_quantization'}_" +\
                            f"mode_{dataset_mode}_outputs.json"
        with open(target_filename, "w") as f:
            json.dump(outputs, f)
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

import copy
import json
import os
import torch

# from sentencepiece import SentencePieceProcessor
from torch.utils.data import Dataset



def relations_to_json(relations, sentence):
    entities_dicts = []
    for idx, rel in enumerate(relations):
        entities_dicts.append({
            "gene": sentence[rel[0]: rel[1]],
            "disease": sentence[rel[2]: rel[3]],
            "relation": rel[4]
        })
    output = json.dumps(entities_dicts)
    return output

REMOVE_ALL_NEGATIVES = False
# INSTRUCTION = """Given a text, extract the gene-disease-relation triplets in a json format."""
INSTRUCTION = """**Extract Triplets**: Identify and extract triplets of relations among an entity pair.
    **Entity Types**: ...TODO fill this in
    **Relation Types**: ...TODO fill this in"""

SYS_PROMPT = """You are a helpful assistant that extracts the list of {"<entity1_type>":<entity1_text>, "<entity2_type>":<entity2_text>", "relation":<relation_type>} triplets from the given text. If no triplets are found, please provide an empty list. """
class DocRedDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, split_name, max_words=None):
        #self.data = json.load(open(dataset_config.data_path))

        if split_name == "train":
            with open(dataset_config.data_path + "train_annotated.json", "r") as f:
                dataset = f.readlines()
        elif split_name == "val":
            with open(dataset_config.data_path + "dev.json", "r") as f:
                dataset = f.readlines()
        elif split_name == "test":
            with open(dataset_config.data_path + "test.json", "r") as f:
                dataset = f.readlines()
        else:
            raise ValueError(f"Invalid split name: {split_name}")
        dataset = json.loads(dataset)
        # dataset is split into sentences I want to treat each sentence as a separate example
        ## TODO continue from this point to fix for the DocRed dataset
        new_dataset = []
        for sample in dataset:
            for idx, sentence in enumerate(sample["sentences"]):
                # strip whitespace from start and end of sentence while counting the number of whitespaces striped from
                # the start of the sentence
                stripped_sentence = sentence.lstrip()
                num_leading_spaces = len(sentence) - len(stripped_sentence)
                entities = self.correct_entity_char_index(sample["ner"][idx], sample["sentences"], idx,
                                                          num_leading_spaces, stripped_sentence)
                relations = self.correct_relation_char_index(sample["relations"][idx], sample["sentences"], idx,
                                                             num_leading_spaces, stripped_sentence)
                new_sample = {
                    "input": sentence.strip(),
                    "output": relations_to_json(relations, sentence.strip()),
                    "doc_key": sample["doc_key"] + f"_sentence_{idx}",
                    "entities": entities
                }
                new_dataset.append(new_sample)
        self.data = new_dataset
        if REMOVE_ALL_NEGATIVES:
            self.data = [item for item in self.data if item["output"] != "[]"]
        # remove data that only contains whitespace
        self.data = [item for item in self.data if item["input"].strip()]
        self.num_negative_examples = len([item for item in self.data if item["output"] == "[]"])
        self.num_positive_examples = len(self.data) - self.num_negative_examples
        self.max_words = max_words
        self.tokenizer = tokenizer
        # self.num_truncated_examples = 0
        # self.longest_input = 0
        # self.input_seen = set()
        self.upweight_minority_class = dataset_config.upweight_minority_class
        self.use_entity_tokens_as_targets = dataset_config.use_entity_tokens_as_targets
        self.entity_special_tokens = dataset_config.entity_special_tokens
        if dataset_config.use_entity_tokens_as_targets:
            if dataset_config.entity_special_tokens:
                self.gene_special_token_id = tokenizer.vocab['<|gene token|>']
                self.disease_special_token_id = tokenizer.vocab['<|disease token|>']
                self.relation_special_token_id = tokenizer.vocab['<|relation token|>']
                self.no_entity_special_token_id = tokenizer.vocab['<|no entity token|>']
            else:
                self.gene_special_token_id = tokenizer.vocab['Ġgene']
                self.disease_special_token_id = tokenizer.vocab['Ġdisease']
                self.relation_special_token_id = tokenizer.vocab['Ġrelation']
                self.no_entity_special_token_id = tokenizer.vocab['Ġnull']
        self.bidirectional_attention_in_entity_tokens = dataset_config.bidirectional_attention_in_entity_tokens
        self.shift_entity_tokens = dataset_config.shift_entity_tokens
        # save the data to a {split_name}_gold.json file
        with open(dataset_config.data_path + f"{split_name}_gold_rel.txt", "w") as f:
            for item in self.data:
                f.write(json.dumps(item) + "\n")

    def get_weights(self, positive_factor=1, negative_factor=1):
        # return the weights for each example
        weights = []
        positive_weight = len(self.data) / self.num_positive_examples * positive_factor
        negative_weight = len(self.data) / self.num_negative_examples * negative_factor
        for item in self.data:
            if item["output"] != "[]":
                weights.append(positive_weight)
            else:
                weights.append(negative_weight)
        return weights

    @staticmethod
    def correct_overlap(corrected_entities, stripped_sentence):
        found_an_overlap = True
        while found_an_overlap:
            found_an_overlap = False
            entities_to_remove = set()
            additional_entities_to_add = set()
            for i in range(len(corrected_entities)):
                for j in range(i + 1, len(corrected_entities)):
                    if corrected_entities[i][0] <= corrected_entities[j][0] and corrected_entities[i][1] >= corrected_entities[j][1]:
                        found_an_overlap = True
                        if corrected_entities[i][2] == corrected_entities[j][2]:
                            entities_to_remove.add(j)
                        else:
                            # if the text is "CNS neuroblastoma with  FOXR2  activation" then
                            if stripped_sentence[corrected_entities[i][0]: corrected_entities[i][1]] in\
                                    ["CNS neuroblastoma with  FOXR2  activation",
                                     "CNS NB- FOXR2",
                                     "anti-cancer drug resistance",
                                     "target for cancer chemotherapy",
                                     "molecular target for anti-cancer drug development"]:
                                # remove the large entity and split it into two entities left and right of the smaller entity
                                # the smaller entity is "FOXR2"
                                left_entity = [corrected_entities[i][0], corrected_entities[j][0], corrected_entities[i][2]]
                                right_entity = [corrected_entities[j][1], corrected_entities[i][1], corrected_entities[i][2]]
                                if left_entity[0] != left_entity[1]:
                                    # add the entity only if it is not empty
                                    additional_entities_to_add.add(tuple(left_entity))
                                if right_entity[0] != right_entity[1]:
                                    # add the entity only if it is not empty
                                    additional_entities_to_add.add(tuple(right_entity))
                                entities_to_remove.add(i)
                            else:
                                raise ValueError(f"Entities of different types overlap: {corrected_entities[i]} {corrected_entities[j]}")
                    elif corrected_entities[j][0] <= corrected_entities[i][0] and corrected_entities[j][1] >= corrected_entities[i][1]:
                        found_an_overlap = True
                        if corrected_entities[i][2] == corrected_entities[j][2]:
                            entities_to_remove.add(i)
                        else:
                            if stripped_sentence[corrected_entities[i][0]: corrected_entities[i][1]] in\
                                    ["CNS neuroblastoma with  FOXR2  activation",
                                     "CNS NB- FOXR2",
                                     "anti-cancer drug resistance",
                                     "target for cancer chemotherapy",
                                     "molecular target for anti-cancer drug development"]:
                                # remove the large entity and split it into two entities left and right of the smaller entity
                                # the smaller entity is "FOXR2"
                                left_entity = [corrected_entities[j][0], corrected_entities[i][0], corrected_entities[j][2]]
                                right_entity = [corrected_entities[i][1], corrected_entities[j][1], corrected_entities[j][2]]
                                if left_entity[0] != left_entity[1]:
                                    # add the entity only if it is not empty
                                    additional_entities_to_add.add(tuple(left_entity))
                                if right_entity[0] != right_entity[1]:
                                    # add the entity only if it is not empty
                                    additional_entities_to_add.add(tuple(right_entity))
                                entities_to_remove.add(j)
                            raise ValueError(f"Entities of different types overlap: {corrected_entities[i]} {corrected_entities[j]}")
            corrected_entities = [entity for idx, entity in enumerate(corrected_entities) if idx not in entities_to_remove]
            additional_entities_to_add = [list(entity) for entity in additional_entities_to_add]
            corrected_entities.extend(additional_entities_to_add)
        # sort again
        corrected_entities.sort(key=lambda x: (x[0], x[1]))
        return corrected_entities

    @staticmethod
    def remove_trailing_whitespace(entities, stripped_sentence, sentence_idx, num_leading_spaces, offset, sentences):
        # if the corrected entity text ends with whitespace then reduce the end index until it does not end with whitespace.
        # The reason for this is that the tokenizer will match the whitespace at the end of the entity text with
        # the whitespace at the start of the next token
        # and thus, the entity will be tokenized as one token longer than it should be.
        for entity in entities:
            while (entity[1] > entity[0]) \
                    and stripped_sentence[entity[1] - 1].isspace():
                entity[1] -= 1
        return entities

    @staticmethod
    def relation_remove_trailing_whitespace(relations, stripped_sentence, sentence_idx, num_leading_spaces, offset, sentences):
        # if the corrected entity text ends with whitespace then reduce the end index until it does not end with whitespace.
        # The reason for this is that the tokenizer will match the whitespace at the end of the entity text with
        # the whitespace at the start of the next token
        # and thus, the entity will be tokenized as one token longer than it should be.
        for relation in relations:
            while (relation[3] > relation[2]) \
                    and stripped_sentence[relation[3] - 1].isspace():
                relation[3] -= 1
            while (relation[1] > relation[0]) \
                    and stripped_sentence[relation[1] - 1].isspace():
                relation[1] -= 1
        return relations

    @staticmethod
    def correct_entity_char_index(entities, sentences, sentence_idx, num_leading_spaces, stripped_sentence):
        # correct entity character indexes to be relative to the sentence and not the whole text
        # and remove leading spaces
        offset = sum([len(sentence) for sentence in sentences[:sentence_idx]]) + num_leading_spaces
        corrected_entities = []
        for entity in entities:
            if type(entity[0]) == list:
                # Entities that have list of indexes must be split into multiple entities
                for idx in range(len(entity[0])):
                    corrected_entities.append([entity[0][idx] - offset,
                                               entity[1][idx] - offset,
                                               entity[2]])
            else:
                corrected_entities.append([entity[0] - offset,
                                           entity[1] - offset,
                                           entity[2]])
        # remove duplicates
        corrected_entities = list(set(tuple(entity) for entity in corrected_entities))
        corrected_entities = [list(entity) for entity in corrected_entities]
        # sort in increasing order of start index and if start indexes are equal then sort in increasing order of end index
        corrected_entities.sort(key=lambda x: (x[0], x[1]))
        # remove any entities that are completely within another entity of the same type. If the types are different then raise an error
        corrected_entities = BioTriplexRelDataset.correct_overlap(corrected_entities, stripped_sentence)
        # remove any trailing whitespace
        corrected_entities = BioTriplexRelDataset.remove_trailing_whitespace(corrected_entities, stripped_sentence,
                                                                          sentence_idx, num_leading_spaces, offset, sentences)
        return corrected_entities

    def correct_relation_char_index(self, relations, sentences, sentence_idx, num_leading_spaces, stripped_sentence):
        # correct relation character indexes to be relative to the sentence and not the whole text
        # and remove leading spaces
        offset = sum([len(sentence) for sentence in sentences[:sentence_idx]]) + num_leading_spaces
        corrected_relations = []
        for relation in relations:
            if type(relation[0]) == list:
                if type(relation[2]) == list:
                    # Relations that have list of indexes must be split into multiple relations
                    for idx in range(len(relation[0])):
                        corrected_relations.append([relation[0][idx] - offset,
                                                   relation[1][idx] - offset,
                                                   relation[2][idx] - offset,
                                                   relation[3][idx] - offset,
                                                   relation[4]])
                else:
                    for idx in range(len(relation[0])):
                        corrected_relations.append([relation[0][idx] - offset,
                                                   relation[1][idx] - offset,
                                                   relation[2] - offset,
                                                   relation[3] - offset,
                                                   relation[4]])
            else:
                if type(relation[2]) == list:
                    for idx in range(len(relation[2])):
                        corrected_relations.append([relation[0] - offset,
                                                   relation[1] - offset,
                                                   relation[2][idx] - offset,
                                                   relation[3][idx] - offset,
                                                   relation[4]])
                else:
                    corrected_relations.append([relation[0] - offset,
                                               relation[1] - offset,
                                               relation[2] - offset,
                                               relation[3] - offset,
                                               relation[4]])
        # remove duplicates
        corrected_relations = list(set(tuple(relation) for relation in corrected_relations))
        corrected_relations = [list(relation) for relation in corrected_relations]
        # sort in increasing order of start index and if start indexes are equal then sort in increasing order of end index
        corrected_relations.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
        # commented out because it is not needed for the relation indexes
        # # remove any entities that are completely within another entity of the same type. If the types are different then raise an error
        # corrected_relations = BioTriplexRelDataset.correct_overlap(corrected_relations, stripped_sentence)
        # remove any trailing whitespace
        corrected_relations = BioTriplexRelDataset.relation_remove_trailing_whitespace(corrected_relations, stripped_sentence,
                                                                            sentence_idx, num_leading_spaces, offset, sentences)
        return corrected_relations


    def get_all_input_prompts(self, bidirectional=False):
        prompts = {}
        for item in self.data:
            prompt = self.input_to_prompt(item["input"])
            if bidirectional:
                bidirectional_region_start = len(prompt[0])
                bidirectional_region_end = len(prompt[0]) + len(prompt[1])
                prompts[item["doc_key"]] = {
                    "prompt": "".join(prompt),
                    "bidirectional_region_start": bidirectional_region_start,
                    "bidirectional_region_end": bidirectional_region_end
                }
            else:
                prompts[item["doc_key"]] = "".join(prompt)
        return prompts

    def input_to_prompt(self, input_text):
        # prompt = f"### Instruction:\n{INSTRUCTION}\n\n### Input:\n{input_text}\n\n### Response:\n"
        prompt_prefix = f"<|start_header_id|>system<|end_header_id|>{SYS_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>" +\
            f"### Instruction:\n{INSTRUCTION}\n### Input:\n"
        prompt_input = input_text + "\n\n"
        prompt_suffix = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        return prompt_prefix, prompt_input, prompt_suffix

    @staticmethod
    def get_entity_indexes(entities, prompt_offsets_mapping, index_offset=0, debug=False):
        genes_indexes = []
        diseases_indexes = []
        relations_indexes = []
        entity_idx = 0
        entities_matched = set()
        for idx, (start, end) in enumerate(prompt_offsets_mapping):
            while entity_idx < len(entities):
                entity = entities[entity_idx]
                start_char, end_char = entity[:2]
                if start <= start_char < end or start < end_char <= end or (start_char < start and end_char > end):
                    # if debug:
                    #     print(start, end, start_char, end_char)
                    #     print(entity_idx, entity)
                    entities_matched.add(entity_idx)
                    if entity[2] == "GENE":
                        genes_indexes.append(idx + index_offset)
                    elif entity[2] == "DISEASE":
                        diseases_indexes.append(idx + index_offset)
                    elif entity[2] == "RELATION":
                        relations_indexes.append(idx + index_offset)
                    else:
                        raise ValueError(f"Invalid entity type: {entity[2]}")
                    break
                elif start_char >= end:
                    break
                elif start >= end_char:
                    entity_idx += 1
                else:
                    raise Exception("Non exchaustive cases")
        if debug:
            print("genes_indexes\n", genes_indexes)
            print("diseases_indexes\n", diseases_indexes)
            print("relations_indexes\n", relations_indexes)
            print("entities\n", entities)
            print("prompt_offsets_mapping\n", prompt_offsets_mapping)
            print("entities idx not matched\n", set(range(len(entities))) - entities_matched)
            print("entities not matched\n", [entities[idx] for idx in set(range(len(entities))) - entities_matched])
        assert len(entities_matched) == len(entities),\
            f"Only {len(entities_matched)} out of {len(entities)} entities found in the prompt"
        return genes_indexes, diseases_indexes, relations_indexes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        # self.input_seen.add(index)

        item = self.data[index]
        debug = False #"4636297_Discussion_sentence_0" #"3979708_Introduction_sentence_0"  # (item["doc_key"] == "9818593_3. Results_sentence_0")
        # prompt = item['input']#f"item['input']\n\n"
        prompt_prefix, prompt_input, prompt_suffix = self.input_to_prompt(item["input"])
        prompt = prompt_prefix + prompt_input + prompt_suffix
        # example = prompt + item["output"]
        example = prompt + "\n### Response:\n" + item["output"]
        if self.upweight_minority_class:
            if item["output"] != "[]":
                weight = POSITIVE_WEIGHT
            else:
                weight = NEGATIVE_WEIGHT
        else:
            weight = 1.0
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(example, dtype=torch.int64)
        # self.longest_input = max(self.longest_input, example.shape[0])
        if self.max_words is not None:
            raise NotImplementedError("max_words is not implemented")
            padding = self.max_words - example.shape[0]
            if padding > 0:
                example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
            elif padding < 0:
                example = example[: self.max_words]
                # self.num_truncated_examples += 1
        labels = copy.deepcopy(example)
        if self.use_entity_tokens_as_targets:
            prompt_prefix = self.tokenizer.encode(prompt_prefix)
            prompt_input = self.tokenizer(prompt_input, add_special_tokens=False, return_offsets_mapping=True)
            prompt_offsets_mapping = prompt_input["offset_mapping"]
            prompt_input = prompt_input["input_ids"]
            prompt_suffix = self.tokenizer.encode(prompt_suffix, add_special_tokens=False)
            labels[:len(prompt_prefix)] = -1
            labels[len(prompt_prefix): len(prompt_prefix) + len(prompt_input)] = self.no_entity_special_token_id
            bidirectional_region_start = len(prompt_prefix)
            bidirectional_region_end = len(prompt_prefix) + len(prompt_input)
            if debug:
                # print all entity spans:
                for entity in item["entities"]:
                    print(entity[0], entity[1], entity[2], f"entity_span: {item['input'][entity[0]: entity[1]]}end")
            genes_indexes, diseases_indexes, relations_indexes = self.get_entity_indexes(item["entities"],
                                                                                         prompt_offsets_mapping,
                                                                                         index_offset=len(prompt_prefix),
                                                                                         debug=debug)
            assert len(prompt_prefix) + len(prompt_input) + len(prompt_suffix) == len(self.tokenizer.encode(prompt)) # TODO remove this assert
            if self.shift_entity_tokens:
                # shift the entity tokens to the right by one so that the model can predict the entity token
                # at the start of the entity span
                genes_indexes = [idx + 1 for idx in genes_indexes]
                diseases_indexes = [idx + 1 for idx in diseases_indexes]
                relations_indexes = [idx + 1 for idx in relations_indexes]
            labels[genes_indexes] = self.gene_special_token_id
            labels[diseases_indexes] = self.disease_special_token_id
            labels[relations_indexes] = self.relation_special_token_id
            labels[len(prompt_prefix) + len(prompt_input): len(prompt_prefix) + len(prompt_input) + len(prompt_suffix)] = -1
        else:
            prompt = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.int64)
            labels[:len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX
        example_mask = example_mask.float()
        # label_mask = label_mask.float()
        # example[example == -100] = self.tokenizer.pad_token_id
        # labels[labels == -100] = self.tokenizer.pad_token_id

        if debug:
            # print each word with its label and token id if labels != -100
            for i in range(len(self.tokenizer.encode(prompt))):
                if labels[i].item() not in [-1, -100, 128259]:
                    print(self.tokenizer.decode(example[i].item()), labels[i].item(), example[i].item())
            print(item["doc_key"])
            for entity in item["entities"]:
                print(item["input"][entity[0]: entity[1]], entity[2], "entity_span: ", item["input"][entity[0]: entity[1]])

        assert len(example) == len(labels)
        # print("input_ids:", example.tolist())
        # print("labels:", labels.tolist())
        # print("attention_mask:", example_mask.tolist())
        if self.bidirectional_attention_in_entity_tokens and self.use_entity_tokens_as_targets:
            # allows for bidirectional attention in entity tokens by returning the start and end of the entity tokens
            # in the input_ids tensor
            return {
                "input_ids": example.tolist(),
                "labels": labels.tolist(),
                "attention_mask": example_mask.tolist(),
                "weight": weight,
                "bidirectional_region_start": bidirectional_region_start,
                "bidirectional_region_end": bidirectional_region_end
                # "doc_key": item["doc_key"],
                # "label_mask": label_mask
            }
        return {
            "input_ids": example.tolist(),
            "labels": labels.tolist(),
            "attention_mask": example_mask.tolist(),
            "weight": weight
            # "doc_key": item["doc_key"],
            # "label_mask": label_mask
        }


if __name__ == "__main__":
    import transformers
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    tokenizer.add_special_tokens({"additional_special_tokens": ["<|gene token|>",
                                                                "<|disease token|>",
                                                                "<|relation token|>",
                                                                "<|no entity token|>"]})
    from llama_recipes.configs.datasets import biotriplex_dataset
    dataset_config = biotriplex_dataset
    dataset_config.use_entity_tokens_as_targets = True
    for mode in "train", "val", "test":
        dataset = BioTriplexRelDataset(dataset_config, tokenizer, mode, max_words=None,)
        # print number of positive and negative examples (with weight 1 and 0.1 respectively)
        num_positive = 0
        num_negative = 0
        for i in range(len(dataset)):
            if dataset[i]["weight"] == POSITIVE_WEIGHT:
                num_positive += 1
            else:
                num_negative += 1
        print("MODE:", mode)
        print(num_positive, num_negative)
        # print len of longest input
        max_len = 0
        for i in range(len(dataset)):
            max_len = max(max_len, len(dataset[i]["input_ids"]))
        print(max_len)

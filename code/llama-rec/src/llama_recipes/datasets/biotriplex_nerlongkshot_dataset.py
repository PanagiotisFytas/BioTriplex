import copy
import json
import os
import torch

# from sentencepiece import SentencePieceProcessor
from torch.utils.data import Dataset
import re


def annotate_words_with_entities(text, entities):
    """
    Annotates words and punctuation in a text with entity types based on priority or 'Null' if no entity.
    Args:
        text (str): The input text.
        entities (list): A list of tuples where each tuple contains the start index,
                         end index, and entity type, e.g., [(start, end, "EntityType")].
    Returns:
        list: A list of tuples where each tuple is (token, entity_type).
              entity_type is 'Null' if the token is not part of an entity.
    """
    # Define priority for entity types
    entity_priority = {"GENE": 1, "DISEASE": 2, "RELATION": 3}
    # Tokenize text into words and punctuation as separate tokens
    tokens = re.finditer(r'\w+|[^\w\s]', text)
    token_offsets = [(match.start(), match.end(), match.group()) for match in tokens]
    # Annotate each token with its entity type
    annotated_tokens = []
    for start, end, token in token_offsets:
        best_entity_type = "NULL"
        best_priority = float('inf')  # Lowest priority number is the highest priority
        for entity_start, entity_end, entity_label in entities:
            # Check if the token overlaps with the entity span
            if start < entity_end and end > entity_start:
                current_priority = entity_priority.get(entity_label.upper(), float('inf'))
                # Update the best entity type if the current one has higher priority
                if current_priority < best_priority:
                    best_entity_type = entity_label
                    best_priority = current_priority

        annotated_tokens.append((token, best_entity_type))
    annotated_tokens_string = "\n".join([f"{token}: {entity}" for token, entity in annotated_tokens])
    return annotated_tokens_string

FEWSHOTS = [
    {"input": "Our finding suggested SETD2 as a potential epigenetic marker in LUAD patients.",
     "output": """
Our: NULL
finding: NULL
suggested: NULL
SETD2: GENE
as: NULL
a: NULL
potential: NULL
epigenetic: RELATION
marker: RELATION
in: NULL
LUAD: DISEASE
patients: NULL
.: NULL"""},
    {"input": "CDKN2BAS1/ANRIL, located in the 9p21 chromosomic region, has been reported in numerous studies as a genetic risk locus for CAD, intracranial aneurysms and diverse cardiometabolic disorders.",
     "output": """
CDKN2BAS1: GENE
/: NULL
ANRIL: GENE
,: NULL
located: NULL
in : NULL
the: NULL
9p21: NULL
chromosomic: NULL
region: NULL
,: NULL
has: NULL
been: NULL
reported: NULL
in: NULL
numerous: NULL
studies: NULL
as: NULL
a: NULL
genetic: RELATION
risk: RELATION
locus: RELATION
for: NULL
CAD: DISEASE
,: NULL
intracranial: NULL
aneurysms: NULL
and: NULL
diverse: NULL
cardiometabolic: NULL
disorders: NULL
.: NULL"""},
    {"input": "Duchenne muscular dystrophy (DMD) is an X-linked inherited neuromuscular disorder due to mutations in the dystrophin gene",
     "output": """
Duchenne: DISEASE
muscular: DISEASE
dystrophy: DISEASE
(: NULL
DMD: GENE
): NULL
is: NULL
an: NULL
X-linked: NULL
inherited: NULL
neuromuscular: NULL
disorder: NULL
due: NULL
to: NULL
mutations: RELATION
in: NULL
the: NULL
dystrophin: GENE
gene: GENE"""},
    {"input": "activation of Wnt signalling is nonetheless thought to play an important role in breast tumorigenesis",
     "output": """
activation: NULL
of: NULL
Wnt: NULL
signalling: NULL
is: NULL
nonetheless: NULL
thought: NULL
to: NULL
play: NULL
an: NULL
important: NULL
role: NULL
in: NULL
breast: NULL
tumorigenesis: NULL"""},
]

POSITIVE_WEIGHT = (153 + 317) / (153 * 2)
NEGATIVE_WEIGHT = (153 + 317) / (317 * 2)
REMOVE_ALL_NEGATIVES = False
# INSTRUCTION = """Given a text, extract the gene-disease-relation triplets in a json format."""
INSTRUCTION = """**Extract Named Entities**: Identify and extract three types of entities in the same sequence as they appear in the text. The entity types are:
   - **GENE**: A human gene name, symbol (e.g., SLC02A1, PCSK5) or synonym.
   - **DISEASE**: A specific human disease or disorder name (e.g., lung adenocarcinoma, coronary artery disease).
   - **RELATION**: The relationship between the gene and the human disease (e.g., associated with, causes, inhibits). These relation types of interest are pathological role, causative activation, causative inhibition, causative mutation, modulator decrease disease, modulator increase disease, biomarker, associated mutation, dysregulation, increased expression, decreased expression, epigenetic marker, therapy resistance, prognostic indicator, negative prognostic marker, positive prognostic marker, therapeutic target, diagnostic tool, genetic susceptibility.
   - **NULL**: If the word does not belong to any of the above entity types.
-Answer in the format: word1: ENTITY_TYPE1, word2: ENTITY_TYPE2, word3: ENTITY_TYPE3, ...
-Each word should be followed by a colon and a space and then the entity type.
-The annotation should be in the same order as the words appear in the text.
-Separate each word and entity type pair with a comma and a space.

Examples:

"""

SYS_PROMPT = """You are a helpful assistant that annotates each word in the text with the entity type it belongs to. Entity types are GENE, DISEASE, RELATION and NULL."""
class BioTriplexNERLongDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, split_name, max_words=None):
        #self.data = json.load(open(dataset_config.data_path))

        if split_name == "train":
            with open(dataset_config.data_path + "train_shorter.txt", "r") as f:
                dataset = f.readlines()
        elif split_name == "val":
            with open(dataset_config.data_path + "val_shorter.txt", "r") as f:
                dataset = f.readlines()
        elif split_name == "test":
            with open(dataset_config.data_path + "test_shorter.txt", "r") as f:
                dataset = f.readlines()
        else:
            raise ValueError(f"Invalid split name: {split_name}")
        dataset = [json.loads(line) for line in dataset]
        # dataset is split into sentences I want to treat each sentence as a separate example
        new_dataset = []
        for sample in dataset:
            for idx, sentence in enumerate(sample["sentences"]):
                # strip whitespace from start and end of sentence while counting the number of whitespaces striped from
                # the start of the sentence
                stripped_sentence = sentence.lstrip()
                num_leading_spaces = len(sentence) - len(stripped_sentence)
                entities = self.correct_entity_char_index(sample["ner"][idx], sample["sentences"], idx,
                                                          num_leading_spaces, stripped_sentence)
                new_sample = {
                    "input": sentence.strip(),
                    "output": annotate_words_with_entities(sentence.strip(), entities),
                    "doc_key": sample["doc_key"] + f"_sentence_{idx}",
                    "entities": entities
                }
                new_dataset.append(new_sample)
        self.data = new_dataset
        if REMOVE_ALL_NEGATIVES:
            self.data = [item for item in self.data if item["entities"] != []]
        # remove data that only contains whitespace
        self.data = [item for item in self.data if item["input"].strip()]
        num_genes = 0
        num_diseases = 0
        num_relations = 0
        self.num_items_without_entities = 0
        for item in self.data:
            if len(item["entities"]) == 0:
                self.num_items_without_entities += 1
            for entity in item["entities"]:
                if entity[2] == "GENE":
                    num_genes += 1
                elif entity[2] == "DISEASE":
                    num_diseases += 1
                elif entity[2] == "RELATION":
                    num_relations += 1
                else:
                    raise ValueError(f"Invalid entity type: {entity[2]}")
        self.num_items_with_entities = len(self.data) - self.num_items_without_entities
        self.total_entities = num_genes + num_diseases + num_relations
        self.weights = {
            "genes": self.total_entities / num_genes / 3,
            "diseases": self.total_entities / num_diseases / 3,
            "relations": self.total_entities / num_relations / 3,
            "empty": len(self.data) / self.num_items_without_entities / 2,
            "non_empty": len(self.data) / self.num_items_with_entities / 2
        }
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
        self.num_of_shots = min(max(dataset_config.num_of_shots, 1), len(FEWSHOTS))
        self.fewshots = FEWSHOTS[:self.num_of_shots]
        with open(dataset_config.data_path + f"{split_name}_gold_nerlong.txt", "w") as f:
            for item in self.data:
                f.write(json.dumps(item) + "\n")

    def get_weights(self, positive_factor=1, negative_factor=1):
        # return the weights for each example as the weighted average of the weights for the genes, diseases and relations
        weights = []
        non_empty_items = []
        for item in self.data:
            weight_sum = 0
            for entity in item["entities"]:
                if entity[2] == "GENE":
                    weight_sum += self.weights["genes"]
                elif entity[2] == "DISEASE":
                    weight_sum += self.weights["diseases"]
                elif entity[2] == "RELATION":
                    weight_sum += self.weights["relations"]
                else:
                    raise ValueError(f"Invalid entity type: {entity[2]}")
            if len(item["entities"]) != 0:
                weights.append(self.weights["non_empty"] * weight_sum / len(item["entities"]))
                non_empty_items.append(True)
            else:
                weights.append(self.weights["empty"])
                non_empty_items.append(False)

        # total_weight_non_empty = sum(weights)
        # total_target_weight_non_empty = self.num_items_with_entities
        # # for all the non_empty items, calculate their final weight as the weighted average of the weights for the genes, diseases and relations
        # for idx, item in enumerate(self.data):
        #     if non_empty_items[idx]:
        #         weights[idx] = weights[idx] / total_weight_non_empty * total_target_weight_non_empty
        #     else:
        #         weights[idx] = 1.0
        return weights

    def debug_get_weights(self):
        print("num_genes:", len([entity for item in self.data for entity in item["entities"] if entity[2] == "GENE"]))
        print("num_diseases:", len([entity for item in self.data for entity in item["entities"] if entity[2] == "DISEASE"]))
        print("num_relations:", len([entity for item in self.data for entity in item["entities"] if entity[2] == "RELATION"]))
        print("num_items_without_entities:", self.num_items_without_entities)
        print("num_items_with_entities:", self.num_items_with_entities)
        print("num_items:", len(self.data))
        print("weights:", self.weights)
        print("get_weights:", self.get_weights())

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
        corrected_entities = BioTriplexNERLongDataset.correct_overlap(corrected_entities, stripped_sentence)
        # remove any trailing whitespace
        corrected_entities = BioTriplexNERLongDataset.remove_trailing_whitespace(corrected_entities, stripped_sentence,
                                                                          sentence_idx, num_leading_spaces, offset, sentences)
        return corrected_entities

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
        few_shot_instruction = "\n".join([f"### Example {idx + 1}:\nInput: \"{item['input']}\"\nOutput: {item['output']}" for idx, item in enumerate(self.fewshots)])
        instruction = INSTRUCTION + few_shot_instruction
        prompt_prefix = f"<|start_header_id|>system<|end_header_id|>{SYS_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>" +\
            f"### Instruction:\n{instruction}\n### Input:\n"
        prompt_input = input_text + "\n\n"
        prompt_suffix = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        return prompt_prefix, prompt_input, prompt_suffix

    def sglang_function(self, max_entities=100, max_tokens_per_entity=50):
        from sglang import function, gen
        # prompt_prefix, prompt_input, prompt_suffix = self.input_to_prompt(item["input"])
        # prompt = prompt_prefix + prompt_input + prompt_suffix
        # example = prompt + "\n### Response:\n" + item["output"]
        @function
        def constrained_prompt(s, prompt):
            input_text = prompt.split("### Input:\n")[1].split("\n\n" + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>")[0]
            words = [input_text[m.start():m.end()] for m in re.finditer(r'\w+|[^\w\s]', input_text)]
            num_words = 0
            s += prompt + "\n### Response:\n"
            for word in words:
                s += f"{word}:" + gen(f"entity_type{num_words}", choices=[" GENE", " DISEASE", " RELATION", " NULL"]) + "\n"
                num_words += 1
            s["num_words"] = num_words
        return constrained_prompt

    # def guidance_function(self, max_tokens_per_entity=50):
    #     import guidance
    #     from guidance import gen, zero_or_more, select
    #     @guidance(stateless=True,)
    #     def constraint_prompt(llm, prompt):
    #         return llm + prompt + "\n### Response:\n" + \
    #                select([
    #                    "[" + zero_or_more(f"""{{'span': "{gen('span', stop='"', max_tokens=50)}", "entity_type": "{select(['GENE', 'DISEASE', 'RELATION'])}"}}, """) + \
    #                    f"""{{'span': "{gen('span', stop='"', max_tokens=50)}", "entity_type": "{select(['GENE', 'DISEASE', 'RELATION'])}"}}]""",
    #                    "[]"
    #                ])
    #     return constraint_prompt

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
            if item["entities"] != []:
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
        dataset = BioTriplexNERLongDataset(dataset_config, tokenizer, mode, max_words=None,)
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

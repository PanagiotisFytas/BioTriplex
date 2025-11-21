import copy
import json
import os
import torch
import random

from torch.utils.data import Dataset


POSITIVE_WEIGHT = (153 + 317) / (153 * 2)
NEGATIVE_WEIGHT = (153 + 317) / (317 * 2)
NUM_CLASSES = 4
REMOVE_ALL_NEGATIVES = False
# INSTRUCTION = """Given a text, extract the gene-disease-relation triplets in a json format."""
def INSTRUCTION(triplet):
    text = (f"What is the relation between the gene (or gene product) {triplet['gene']} and the disease (or phenotypic feature) {triplet['disease']}?"
            f"\n a) Association"
            f"\n b) Negative Correlation"
            f"\n c) Positive Correlation"
            f"\n d) No relation"
            "\nPlease select the correct option by answering a), b), c) or d).")
    return text

FEWSHOTS = [
    {"input": "Hepatocyte nuclear factor-6: associations between genetic variability and type II diabetes and between genetic variability and estimates of insulin secretion.\n\nThe transcription factor hepatocyte nuclear factor (HNF)-6 is an upstream regulator of several genes involved in the pathogenesis of maturity-onset diabetes of the young. We therefore tested the hypothesis that variability in the HNF-6 gene is associated with subsets of Type II (non-insulin-dependent) diabetes mellitus and estimates of insulin secretion in glucose tolerant subjects.   We cloned the coding region as well as the intron-exon boundaries of the HNF-6 gene. We then examined them on genomic DNA in six MODY probands without mutations in the MODY1, MODY3 and MODY4 genes and in 54 patients with late-onset Type II diabetes by combined single strand conformational polymorphism-heteroduplex analysis followed by direct sequencing of identified variants. An identified missense variant was examined in association studies and genotype-phenotype studies.   We identified two silent and one missense (Pro75 Ala) variant. In an association study the allelic frequency of the Pro75Ala polymorphism was 3.2% (95% confidence interval, 1.9-4.5) in 330 patients with Type II diabetes mellitus compared with 4.2% (2.4-6.0) in 238 age-matched glucose tolerant control subjects. Moreover, in studies of 238 middle-aged glucose tolerant subjects, of 226 glucose tolerant offspring of Type II diabetic patients and of 367 young healthy subjects, the carriers of the polymorphism did not differ from non-carriers in glucose induced serum insulin or C-peptide responses.   Mutations in the coding region of the HNF-6 gene are not associated with Type II diabetes or with changes in insulin responses to glucose among the Caucasians examined.",
     "question": "What is the relation between the gene (or gene product) Hepatocyte nuclear factor-6 and the disease (or phenotypic feature) type II diabetes?",
     "output": "a)",
     "output_gen": "a)"},
    {"input": "Hepatocyte nuclear factor-6: associations between genetic variability and type II diabetes and between genetic variability and estimates of insulin secretion.\n\nThe transcription factor hepatocyte nuclear factor (HNF)-6 is an upstream regulator of several genes involved in the pathogenesis of maturity-onset diabetes of the young. We therefore tested the hypothesis that variability in the HNF-6 gene is associated with subsets of Type II (non-insulin-dependent) diabetes mellitus and estimates of insulin secretion in glucose tolerant subjects.   We cloned the coding region as well as the intron-exon boundaries of the HNF-6 gene. We then examined them on genomic DNA in six MODY probands without mutations in the MODY1, MODY3 and MODY4 genes and in 54 patients with late-onset Type II diabetes by combined single strand conformational polymorphism-heteroduplex analysis followed by direct sequencing of identified variants. An identified missense variant was examined in association studies and genotype-phenotype studies.   We identified two silent and one missense (Pro75 Ala) variant. In an association study the allelic frequency of the Pro75Ala polymorphism was 3.2% (95% confidence interval, 1.9-4.5) in 330 patients with Type II diabetes mellitus compared with 4.2% (2.4-6.0) in 238 age-matched glucose tolerant control subjects. Moreover, in studies of 238 middle-aged glucose tolerant subjects, of 226 glucose tolerant offspring of Type II diabetic patients and of 367 young healthy subjects, the carriers of the polymorphism did not differ from non-carriers in glucose induced serum insulin or C-peptide responses.   Mutations in the coding region of the HNF-6 gene are not associated with Type II diabetes or with changes in insulin responses to glucose among the Caucasians examined.",
     "question": "What is the relation between the gene (or gene product) insulin and the disease (or phenotypic feature) type II diabetes?",
     "output": "d)",
     "output_gen": "d)"},
    {"input": "Is the European spatial distribution of the HIV-1-resistant CCR5-Delta32 allele formed by a breakdown of the pathocenosis due to the historical Roman expansion?\n\nWe studied the possible effects of the expansion of ancient Mediterranean civilizations during the five centuries before and after Christ on the European distribution of the mutant allele for the chemokine receptor gene CCR5 which has a 32-bp deletion (CCR5-Delta32). There is a strong evidence for the unitary origin of the CCR5-Delta32 mutation, this it is found principally in Europe and Western Asia, with generally a north-south downhill cline frequency. Homozygous carriers of this mutation show a resistance to HIV-1 infection and a slower progression towards AIDS. However, HIV has clearly emerged too recently to have been the selective force on CCR5. Our analyses showed strong negative correlations in Europe between the allele frequency and two historical parameters, i.e. the first colonization dates by the great ancient Mediterranean civilizations, and the distances from the Northern frontiers of the Roman Empire in its greatest expansion. Moreover, other studies have shown that the deletion frequencies in both German Bronze Age and Swedish Neolithic populations were similar to those found in the corresponding modern populations, and this deletion has been found in ancient DNA of around 7000 years ago, suggesting that in the past, the deletion frequency could have been relatively high in European populations. In addition, in West Nile virus pathogenesis, CCR5 plays an antimicrobial role showing that host genetic factors are highly pathogen-specific. Our results added to all these previous data suggest that the actual European allele frequency distribution might not be due to genes spreading, but to a negative selection resulting in the spread of pathogens principally during Roman expansion. Indeed, as gene flows from colonizers to European native populations were extremely low, the mutational changes might be associated with vulnerability to imported infections. To date, the nature of the parasites remains unknown; however, zoonoses could be incriminated.",
     "question": "What is the relation between the gene (or gene product) CCR5 and the disease (or phenotypic feature) AIDS?",
     "output": "a)",
     "output_gen": "a)"},
    {"input": "Is the European spatial distribution of the HIV-1-resistant CCR5-Delta32 allele formed by a breakdown of the pathocenosis due to the historical Roman expansion?\n\nWe studied the possible effects of the expansion of ancient Mediterranean civilizations during the five centuries before and after Christ on the European distribution of the mutant allele for the chemokine receptor gene CCR5 which has a 32-bp deletion (CCR5-Delta32). There is a strong evidence for the unitary origin of the CCR5-Delta32 mutation, this it is found principally in Europe and Western Asia, with generally a north-south downhill cline frequency. Homozygous carriers of this mutation show a resistance to HIV-1 infection and a slower progression towards AIDS. However, HIV has clearly emerged too recently to have been the selective force on CCR5. Our analyses showed strong negative correlations in Europe between the allele frequency and two historical parameters, i.e. the first colonization dates by the great ancient Mediterranean civilizations, and the distances from the Northern frontiers of the Roman Empire in its greatest expansion. Moreover, other studies have shown that the deletion frequencies in both German Bronze Age and Swedish Neolithic populations were similar to those found in the corresponding modern populations, and this deletion has been found in ancient DNA of around 7000 years ago, suggesting that in the past, the deletion frequency could have been relatively high in European populations. In addition, in West Nile virus pathogenesis, CCR5 plays an antimicrobial role showing that host genetic factors are highly pathogen-specific. Our results added to all these previous data suggest that the actual European allele frequency distribution might not be due to genes spreading, but to a negative selection resulting in the spread of pathogens principally during Roman expansion. Indeed, as gene flows from colonizers to European native populations were extremely low, the mutational changes might be associated with vulnerability to imported infections. To date, the nature of the parasites remains unknown; however, zoonoses could be incriminated.",
     "question": "What is the relation between the gene (or gene product) CCR5 and the disease (or phenotypic feature) infections?",
     "output": "d)",
     "output_gen": "d)"},
    {"input": "Permeability, ultrastructural changes, and distribution of novel proteins in the glomerular barrier in early puromycin aminonucleoside nephrosis.\n\nBACKGROUND/AIMS: It is still unclear what happens in the glomerulus when proteinuria starts. Using puromycin aminonucleoside nephrosis (PAN) rats, we studied early ultrastructural and permeability changes in relation to the expression of the podocyte-associated molecules nephrin, a-actinin, dendrin, and plekhh2, the last two of which were only recently discovered in podocytes. METHODS: Using immune stainings, semiquantitative measurement was performed under the electron microscope. Permeability was assessed using isolated kidney perfusion with tracers. Possible effects of ACE inhibition were tested. RESULTS: By day 2, some patchy foot process effacement, but no proteinuria, appeared. The amount of nephrin was reduced in both diseased and normal areas. The other proteins showed few changes, which were limited to diseased areas. By day 4, foot process effacement was complete and proteinuria appeared in parallel with signs of size barrier damage. Nephrin decreased further, while dendrin and plekhh2 also decreased but a-actinin remained unchanged. ACE inhibition had no significant protective effect. CONCLUSIONS: PAN glomeruli already showed significant pathology by day 4, despite relatively mild proteinuria. This was preceded by altered nephrin expression, supporting its pivotal role in podocyte morphology. The novel proteins dendrin and plekhh2 were both reduced, suggesting roles in PAN, whereas a-actinin was unchanged.",
     "question": "What is the relation between the gene (or gene product) plekhh2 and the disease (or phenotypic feature) PAN?",
     "output": "a)",
     "output_gen": "a)"},
]

def fewshot_to_instruction(fewshot, general_relations=False, group_relations=False):
    text = (f"{fewshot['input']}"
        f"{fewshot['question']}"
        f"\n a) Association"
        f"\n b) Negative Correlation"
        f"\n c) Positive Correlation"
        f"\n d) No relation"
        "\nPlease select the correct option by answering a), b), c) or d).")
    return text


def triplet_to_underscore_sep(triplet):
    return "_".join(["gene", triplet["gene"], "disease", triplet["disease"], "relation", triplet["relation"]])

def triplet_to_answer(triplet):
    if triplet["relation"] == "No Relation":
        return "d)"
    elif triplet["relation"] == "Positive_Correlation":
        return "c)"
    elif triplet["relation"] == "Negative_Correlation":
        return "b)"
    elif triplet["relation"] == "Association":
        return "a)"
    else:
        raise ValueError(f"Invalid relation: {triplet['relation']}")

SYS_PROMPT = """You are a helpful assistant that answers questions about the relation between genes and diseases by answering a), b), c) or d) and nothing else."""

class BioRedQADataset(Dataset):
    def __init__(self, dataset_config, tokenizer, split_name, max_words=None):
        #self.data = json.load(open(dataset_config.data_path))
        self.relation_types = set()
        if split_name == "train":
            with open(dataset_config.data_path + "Train.BioC.JSON", "r") as f:
                dataset = json.load(f)
        elif split_name == "val":
            with open(dataset_config.data_path + "Dev.BioC.JSON", "r") as f:
                dataset = json.load(f)
        elif split_name == "test":
            with open(dataset_config.data_path + "Test.BioC.JSON", "r") as f:
                dataset = json.load(f)
        else:
            raise ValueError(f"Invalid split name: {split_name}")
        # dataset is split into sentences I want to treat each sentence as a separate example
        new_dataset = []
        for sample in dataset["documents"]:
            assert len(sample["passages"]) == 2
            # sample 1 is the title and sample 2 is the abstract
            text = sample["passages"][0]["text"] + "\n\n" + sample["passages"][1]["text"]
            extra_offset = len("\n\n")
            assert sample["passages"][0]["offset"] == 0
            assert sample["passages"][1]["offset"] == len(sample["passages"][0]["text"]) + 1
            entities0 = self.correct_entity_char_index(sample["passages"][0]["annotations"], text)
            entities1 = self.correct_entity_char_index(sample["passages"][1]["annotations"], text,
                                                       extra_offset=extra_offset-1)
            entities = entities0 + entities1
            relations = self.correct_relations(sample["relations"], sample["passages"][0]["annotations"] + sample["passages"][1]["annotations"], return_neg_relations=(not REMOVE_ALL_NEGATIVES))
            for relation in relations:
                new_sample = {
                    "input": text,
                    "output": triplet_to_answer(relation),
                    "doc_key": sample["id"] + "_" + triplet_to_underscore_sep(relation),
                    "entities": entities,
                    "relation": relation
                }
                new_dataset.append(new_sample)
        self.data = new_dataset
        # remove data that only contains whitespace
        self.data = [item for item in self.data if item["input"].strip()]
        self.num_samples_per_relation = {}
        for item in self.data:
            self.num_samples_per_relation[item["output"]] = self.num_samples_per_relation.get(item["output"], 0) + 1
        self.max_words = max_words
        self.tokenizer = tokenizer
        # self.num_truncated_examples = 0
        # self.longest_input = 0
        # self.input_seen = set()
        self.upweight_minority_class = dataset_config.upweight_minority_class
        self.use_entity_tokens_as_targets = dataset_config.use_entity_tokens_as_targets
        self.entity_special_tokens = dataset_config.entity_special_tokens
        if tokenizer is not None:
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
        self.num_of_shots = min(max(dataset_config.num_of_shots, 0), len(FEWSHOTS))
        self.fewshots = FEWSHOTS[:self.num_of_shots]
        with open(dataset_config.data_path + f"{split_name}_gold_biored_qa.txt", "w") as f:
            for item in self.data:
                f.write(json.dumps(item) + "\n")

    def get_weights(self, positive_factor=1, negative_factor=1):
        # return the weights for each example
        weights = []
        weights_per_relation = {}
        for output in self.num_samples_per_relation:
            if output != "d)":
                weights_per_relation[output] = len(self.data) / self.num_samples_per_relation[output] * positive_factor\
                                               / len(self.num_samples_per_relation)
            else:
                weights_per_relation[output] = len(self.data) / self.num_samples_per_relation[output] * negative_factor\
                                               / len(self.num_samples_per_relation)
        for item in self.data:
            weights.append(weights_per_relation[item["output"]])
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
    def remove_trailing_whitespace(entities, stripped_sentence,):
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
    def relation_remove_trailing_whitespace(relations, stripped_sentence):
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
    def correct_entity_char_index(entities, sentence, extra_offset=0):
        # correct entity character indexes to be relative to the sentence and not the whole text
        corrected_entities = []
        for entity in entities:
            if entity["infons"]["type"] not in ["GeneOrGeneProduct", "DiseaseOrPhenotypicFeature"]:
                continue
            type = "GENE" if entity["infons"]["type"] == "GeneOrGeneProduct" else "DISEASE"
            for location in entity["locations"]:
                corrected_entities.append([location["offset"] + extra_offset,
                                           location["offset"] + location["length"] + extra_offset,
                                           type,])
                assert entity["text"] == sentence[location["offset"] + extra_offset: location["offset"] + location["length"] + extra_offset]
        # remove any leading spaces
        for entity in corrected_entities:
            while (entity[1] > entity[0]) \
                    and sentence[entity[0]].isspace():
                entity[0] += 1
            while (entity[1] > entity[0]) \
                    and sentence[entity[1] - 1].isspace():
                entity[1] -= 1
            assert entity[0] < entity[1]
        # remove duplicates
        corrected_entities = list(set(tuple(entity) for entity in corrected_entities))
        corrected_entities = [list(entity) for entity in corrected_entities]
        # sort in increasing order of start index and if start indexes are equal then sort in increasing order of end index
        corrected_entities.sort(key=lambda x: (x[0], x[1]))
        # remove any trailing whitespace
        corrected_entities = BioRedQADataset.remove_trailing_whitespace(corrected_entities, sentence)

        # remove any entities that are completely within another entity of the same type. If the types are different then raise an error
        corrected_entities = BioRedQADataset.correct_overlap(corrected_entities, sentence)
        return corrected_entities

    def correct_relations(self, relations, entity_annotations, return_neg_relations=False):
        corrected_relations = []
        entities_dicts = {}
        for annotations in entity_annotations:
            if annotations["infons"]["type"] not in ["GeneOrGeneProduct", "DiseaseOrPhenotypicFeature"]:
                continue
            # remove any leading or trailing whitespace
            id = annotations["infons"]["identifier"]
            annotation_text = annotations["text"]
            start = min(annotations["locations"][idx]["offset"] for idx in range(len(annotations["locations"])))
            while annotation_text and annotation_text[0].isspace():
                annotation_text = annotation_text[1:]
            while annotation_text and annotation_text[-1].isspace():
                annotation_text = annotation_text[:-1]
            annotation_type = "GENE" if annotations["infons"]["type"] == "GeneOrGeneProduct" else "DISEASE"
            entities_dicts[id] = entities_dicts.get(id, []) + [[annotation_text, annotation_type, start]]
        # remove duplicates
        for id, entities in entities_dicts.items():
            entities_dicts[id] = list(set(tuple(entity) for entity in entities))
            entities_dicts[id] = [list(entity) for entity in entities_dicts[id]]

        for relation in relations:
            if relation["infons"]["entity1"] in entities_dicts and relation["infons"]["entity2"] in entities_dicts:
                # consider only gene-disease and disease-gene relations
                if (entities_dicts[relation["infons"]["entity1"]][0][1] == "GENE" and \
                        entities_dicts[relation["infons"]["entity2"]][0][1] == "DISEASE"):
                    gene_texts = [entities_dicts[relation["infons"]["entity1"]][idx][0] for idx in range(len(entities_dicts[relation["infons"]["entity1"]]))]
                    gene_starts = [entities_dicts[relation["infons"]["entity1"]][idx][2] for idx in range(len(entities_dicts[relation["infons"]["entity1"]]))]
                    disease_texts = [entities_dicts[relation["infons"]["entity2"]][idx][0] for idx in range(len(entities_dicts[relation["infons"]["entity2"]]))]
                    disease_starts = [entities_dicts[relation["infons"]["entity2"]][idx][2] for idx in range(len(entities_dicts[relation["infons"]["entity2"]]))]
                elif (entities_dicts[relation["infons"]["entity2"]][0][1] == "GENE" and \
                        entities_dicts[relation["infons"]["entity1"]][0][1] == "DISEASE"):
                    gene_texts = [entities_dicts[relation["infons"]["entity2"]][idx][0] for idx in range(len(entities_dicts[relation["infons"]["entity2"]]))]
                    gene_starts = [entities_dicts[relation["infons"]["entity2"]][idx][2] for idx in range(len(entities_dicts[relation["infons"]["entity2"]]))]
                    disease_texts = [entities_dicts[relation["infons"]["entity1"]][idx][0] for idx in range(len(entities_dicts[relation["infons"]["entity1"]]))]
                    disease_starts = [entities_dicts[relation["infons"]["entity1"]][idx][2] for idx in range(len(entities_dicts[relation["infons"]["entity1"]]))]
                else:
                    continue
                for gene_text, gene_start in zip(gene_texts, gene_starts):
                    for disease_text, disease_start in zip(disease_texts, disease_starts):
                        corrected_relations.append([gene_text, disease_text, relation["infons"]["type"], min(gene_start, disease_start), max(gene_start, disease_start)])
        if return_neg_relations:
            relation_ids = [(relation["infons"]["entity1"], relation["infons"]["entity2"]) for relation in relations]
            for gene_id, gene_entities in entities_dicts.items():
                if gene_entities[0][1] != "GENE":
                    continue
                for disease_id, disease_entities in entities_dicts.items():
                    if disease_entities[0][1] != "DISEASE":
                        continue
                    if (gene_id, disease_id) in relation_ids or (disease_id, gene_id) in relation_ids:
                        continue
                    gene_texts = [gene_entities[idx][0] for idx in range(len(gene_entities))]
                    gene_starts = [gene_entities[idx][2] for idx in range(len(gene_entities))]
                    disease_texts = [disease_entities[idx][0] for idx in range(len(disease_entities))]
                    disease_starts = [disease_entities[idx][2] for idx in range(len(disease_entities))]
                    for gene_text, gene_start in zip(gene_texts, gene_starts):
                        for disease_text, disease_start in zip(disease_texts, disease_starts):
                            corrected_relations.append([gene_text, disease_text, "No Relation", min(gene_start, disease_start), max(gene_start, disease_start)])
        # sort based on the start index
        corrected_relations.sort(key=lambda x: (x[3], x[4]))
        # remove start
        corrected_relations = [relation[:3] for relation in corrected_relations]
        # remove duplicates (of the form [gene, disease, relation]) but keep the first one and dont break the order
        deduplicated_relations = []
        seen_relations = set()
        for relation in corrected_relations:
            self.relation_types.add(relation[2])
            if tuple(relation) not in seen_relations:
                deduplicated_relations.append(relation)
                seen_relations.add(tuple(relation))
        # make into a json string of format [{"gene": gene_text, "disease": disease_text, "relation": relation_type}, ...]
        corrected_relations = [{"gene": relation[0], "disease": relation[1], "relation": relation[2]} for relation in deduplicated_relations]
        # corrected_relations = json.dumps(corrected_relations)
        return corrected_relations


    def get_all_input_prompts(self, bidirectional=False):
        prompts = {}
        for item in self.data:
            prompt = self.input_to_prompt(item["input"], item["relation"])
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

    def openai_get_all_input_prompts(self):
        prompts = {}
        for item in self.data:
            prompt = self.openai_input_to_prompt(item["input"], item["relation"])
            prompts[item["doc_key"]] = prompt
        return prompts

    def openai_input_to_prompt(self, input_text, triplet):
        instr = INSTRUCTION(triplet)
        sys_prompt = SYS_PROMPT

        messages = []

        # Add system message
        messages.append({"role": "system", "content": sys_prompt})

        # Add few-shot examples
        for fewshot in self.fewshots:
            fewshot_instruction = fewshot_to_instruction(fewshot)
            user_msg = f"### Instruction:\n{fewshot_instruction}\n### Input:\n{fewshot['input']}"
            assistant_msg = f"### Response:\n{fewshot['output']}"
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})

        # Add final user query
        user_query = f"### Instruction:\n{instr}\n### Input:\n{input_text}"
        messages.append({"role": "user", "content": user_query})

        return messages


    def input_to_prompt(self, input_text, triplet):
        # prompt = f"### Instruction:\n{INSTRUCTION}\n\n### Input:\n{input_text}\n\n### Response:\n"
        instr = INSTRUCTION(triplet)
        sys_prompt = SYS_PROMPT
        fewshot_prompts = ""
        for fewshot in self.fewshots:
            fewshot_prompts += "<|start_header_id|>user<|end_header_id|>"
            fewshot_prompts += f"### Instruction:\n{fewshot_to_instruction(fewshot)}"
            fewshot_prompts += f"\n### Input:\n{fewshot['input']}"
            fewshot_prompts += "\n\n" + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
            fewshot_prompts += f"\n### Response:\n{fewshot['output']}"
            fewshot_prompts += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

        prompt_prefix = f"<|start_header_id|>system<|end_header_id|>{sys_prompt}<|eot_id|>{fewshot_prompts}<|start_header_id|>user<|end_header_id|>" +\
            f"### Instruction:\n{instr}\n### Input:\n"
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
        prompt_prefix, prompt_input, prompt_suffix = self.input_to_prompt(item["input"], item["relation"])
        prompt = prompt_prefix + prompt_input + prompt_suffix
        # example = prompt + item["output"]
        example = prompt + "\n### Response:\n" + item["output"]
        if self.upweight_minority_class:
            if item["output"] != "d)":
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
    from llama_recipes.configs.datasets import biored_dataset
    dataset_config = biored_dataset
    dataset_config.use_entity_tokens_as_targets = True
    for mode in "test", "train", "val":
        dataset = BioRedQADataset(dataset_config, tokenizer, mode, max_words=None,)
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
        print("Relation Types:\n", dataset.relation_types)
        # print(tokenizer.decode(dataset[0]["input_ids"]))

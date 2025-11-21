import copy
import json
import torch
import random
from torch.utils.data import Dataset


POSITIVE_WEIGHT = (153 + 317) / (153 * 2)
NEGATIVE_WEIGHT = (153 + 317) / (317 * 2)
NUM_CLASSES = 4
REMOVE_ALL_NEGATIVES = False
# INSTRUCTION = """Given a text, extract the gene-disease-relation triplets in a json format."""
RELATIONS = [
    "pathological role", "causative activation", "causative inhibition", "causative mutation",
    "modulator decrease disease", "modulator increase disease", "biomarker", "associated mutation",
    "dysregulation", "increased expression", "decreased expression", "epigenetic marker",
    "therapy resistance", "prognostic indicator", "negative prognostic marker", "positive prognostic marker",
    "therapeutic target", "diagnostic tool", "genetic susceptibility", "no relation", "relation undefined"
]
GENERAL_REL = {
    "disease causing": ["pathological role", "causative activation", "causative inhibition", "causative mutation",
                     "associated mutation"],
    "disease modulating": ["modulator decrease disease", "modulator increase disease", "genetic susceptibility"],
    "gene expression": ["increased expression", "decreased expression", "dysregulation"],
    "diagnosis": ["biomarker", "diagnostic tool", "epigenetic marker", "prognostic indicator",
                  "positive prognostic marker", "negative prognostic marker"],
    "therapy": ["therapy resistance", "therapeutic target"],
    "no relation": ["no relation"],
    "relation undefined": ["relation undefined"]
}
GENERAL_REL_LIST = [
    "disease causing", "disease modulating", "gene expression", "diagnosis", "therapy", "no relation", "relation undefined"
]

# GENERAL_REL = {
#     "pathological": ["pathological role", "causative activation", "causative inhibition", "causative mutation",
#                      "associated mutation"],
#     "modulatory": ["modulator decrease disease", "modulator increase disease", "genetic susceptibility"],
#     "gene expression": ["increased expression", "decreased expression", "dysregulation"],
#     "diagnosis": ["biomarker", "diagnostic tool", "epigenetic marker", "prognostic indicator",
#                   "positive prognostic marker", "negative prognostic marker"],
#     "therapy": ["therapy resistance", "therapeutic target"],
#     "no relation": ["no relation"],
#     "relation undefined": ["relation undefined"]
# }
# GENERAL_REL_LIST = [
#     "pathological", "modulatory", "gene expression", "diagnosis", "therapy", "no relation", "relation undefined"
# ]
NEGATIVE_REL = "no relation"
NEGATIVE_OPTION = f"{chr(ord('a') + RELATIONS.index(NEGATIVE_REL))})"
OPTIONS = [f"{chr(ord('a') + idx)})" for idx, _ in enumerate(RELATIONS)]
OPTIONS = ", ".join(OPTIONS)
OPTIONS = OPTIONS[:-2] + "or " + OPTIONS[-2:]
GENERAL_OPTIONS = [f"{chr(ord('a') + idx)})" for idx, _ in enumerate(GENERAL_REL_LIST)]
GENERAL_OPTIONS = ", ".join(GENERAL_OPTIONS)
GENERAL_OPTIONS = GENERAL_OPTIONS[:-2] + "or " + GENERAL_OPTIONS[-2:]


FEWSHOTS = [
    {"input": "Our finding suggested SETD2 as a potential epigenetic marker in LUAD patients.",
     "question": "What is the relation between the gene SETD2 and the disease LUAD?",
     "output": "l)",
     "output_gen": "d)"},
    {"input": "CDKN2BAS1/ANRIL, located in the 9p21 chromosomic region, has been reported in numerous studies as a genetic risk locus for CAD, intracranial aneurysms and diverse cardiometabolic disorders.",
     "question": "What is the relation between the gene CDKN2BAS1 and the disease CAD?",
     "output": "s)",
     "output_gen": "b)"},
    {"input": "Duchenne muscular dystrophy (DMD) is an X-linked inherited neuromuscular disorder due to mutations in the dystrophin gene",
     "question": "What is the relation between the gene dystrophin and the disease Duchenne muscular dystrophy?",
     "output": "d)",
     "output_gen": "a)"},
    {"input": "The genetic variation of RAS encoding genes, angiotensinogen ( AGT ), angiotensin-1-converting enzyme ( ACE ), and angiotensin II type 1 receptor ( AGTR1 ), were associated with EHT and have been important genes for the association studies in various populations [ 8 – 10 ].  AGT , the natural substrate of RAS, is synthesized in the liver and released into blood circulation. The potential role of  AGT  gene in hypertension was originally explored by Jeunemaitre group through linkage and association study in the causation of human EHT in Utah and French populations [ 4 ].",
     "question": "What is the relation between the gene EHT and the disease hypertension?",
     "output": "s)",
     "output_gen": "f)"},
]

GROUPED_FEWSHOTS = [
    {"input": "Our finding suggested SETD2 as a potential epigenetic marker in LUAD patients.",
     "question": "What are the relations between the gene SETD2 and the disease LUAD that appear in the text?",
     "output": "l)",
     "output_gen": "d)"},
    {"input": "CDKN2BAS1/ANRIL, located in the 9p21 chromosomic region, has been reported in numerous studies as a genetic risk locus for CAD, intracranial aneurysms and diverse cardiometabolic disorders.",
     "question": "What are the relations between the gene CDKN2BAS1 and the disease CAD that appear in the text?",
     "output": "s)",
     "output_gen": "b)"},
    {"input": "Duchenne muscular dystrophy (DMD) is an X-linked inherited neuromuscular disorder due to mutations in the dystrophin gene",
     "question": "What are the relations between the gene dystrophin and the disease Duchenne muscular dystrophy that appear in the text?",
     "output": "d)",
     "output_gen": "a)"},
    {"input": "The genetic variation of RAS encoding genes, angiotensinogen ( AGT ), angiotensin-1-converting enzyme ( ACE ), and angiotensin II type 1 receptor ( AGTR1 ), were associated with EHT and have been important genes for the association studies in various populations [ 8 – 10 ].  AGT , the natural substrate of RAS, is synthesized in the liver and released into blood circulation. The potential role of  AGT  gene in hypertension was originally explored by Jeunemaitre group through linkage and association study in the causation of human EHT in Utah and French populations [ 4 ].",
     "question": "What are the relations between the gene EHT and the disease hypertension that appear in the text?",
     "output": "s)",
     "output_gen": "f)"},
    {"input": "high expression of METTL3, HNRNPA2B1, and YTHDF3 were related to the poor prognosis of osteosarcoma",
     "question": "What are the relations between the gene HNRNPA2B1 and the disease osteosarcoma that appear in the text?",
     "output": "j), o)",
     "output_gen": "c), d)"},
    {"input": "high expression of METTL3, HNRNPA2B1, and YTHDF3 were related to the poor prognosis of osteosarcoma",
     "question": "What are the relations between the gene METTL3 and the disease osteosarcoma that appear in the text?",
     "output": "j), o)",
     "output_gen": "c), d)"},
    {"input": "high expression of METTL3, HNRNPA2B1, and YTHDF3 were related to the poor prognosis of osteosarcoma",
     "question": "What are the relations between the gene YTHDF3 and the disease osteosarcoma that appear in the text?",
     "output": "j), o)",
     "output_gen": "c), d)"},
]


def INSTRUCTION(triplet, general_relations=False, group_relations=False):
    if general_relations:
        relations = GENERAL_REL_LIST
        options = GENERAL_OPTIONS
    else:
        relations = RELATIONS
        options = OPTIONS
    relation_options = "".join([f"\n {chr(ord('a') + idx)}) {relation}" for idx, relation in enumerate(relations)])
    if group_relations:
        text = (f"What are the relations between the gene {triplet['gene']} and the disease {triplet['disease']} that appear in the text?"
                f"{relation_options}"
                f"\nPlease select one or more of the options ({options}) in a comma separated list.")

    else:
        text = (f"What is the relation between the gene {triplet['gene']} and the disease {triplet['disease']}?"
                f"{relation_options}"
                f"\nPlease select the correct option by answering {options} and nothing else.")
    return text

def fewshot_to_instruction(fewshot, general_relations=False, group_relations=False):
    if general_relations:
        relations = GENERAL_REL_LIST
        options = GENERAL_OPTIONS
    else:
        relations = RELATIONS
        options = OPTIONS
    relation_options = "".join([f"\n {chr(ord('a') + idx)}) {relation}" for idx, relation in enumerate(relations)])
    if group_relations:
        text = (f"{fewshot['input']}"
            f"{fewshot['question']}"
            f"{relation_options}"
            f"\nPlease select one or more of the options ({options}) in a comma separated list and nothing else.")
    else:
        text = (f"{fewshot['question']}"
                f"{relation_options}"
                f"\nPlease select the correct option by answering {options} and nothing else.")
    return text

def triplet_to_underscore_sep(triplet, group_relations=False):
    if group_relations:
        return "_".join(["gene", triplet["gene"], "disease", triplet["disease"], "relations", "_".join(triplet["relation"])])
    return "_".join(["gene", triplet["gene"], "disease", triplet["disease"], "relation", triplet["relation"]])


def grouped_triplet_to_answer(triplet, general_relations=False):
    answers = []
    for relation in triplet["relation"]:
        answer = triplet_to_answer({"gene": triplet["gene"], "disease": triplet["disease"], "relation": relation},
                                   general_relations=general_relations)
        answers.append(answer)
    # sort the answers in alphabetical order
    answers.sort()
    return ", ".join(answers)

def triplet_to_answer(triplet, general_relations=False):
    try:
        if general_relations:
            idx_rel = GENERAL_REL_LIST.index(triplet.get("relation"))
            # general_rel_list is precomputed now
            # idx = 0
            # keys = GENERAL_REL_LIST
            # for key in keys:
            #     if triplet["relation"].lower() in GENERAL_REL[key]:
            #         idx_rel = idx
            #         break
            #     idx += 1
        else:
            idx_rel = RELATIONS.index(triplet["relation"].lower())
    except ValueError:
        raise ValueError(f"Invalid relation: {triplet['relation']}")
    return f"{chr(ord('a') + idx_rel)})"


def link_is_equal(relation, gene, disease, sentence):
    eq = (sentence[relation[0]: relation[1]].strip() == sentence[gene[0]: gene[1]].strip()) and\
         (sentence[relation[2]: relation[3]].strip() == sentence[disease[0]: disease[1]].strip())
    return eq

def triplet_has_equal_gene_and_disease(triplet, gene, disease):
    eq = (triplet["gene"].lower().strip() == gene.lower().strip()) and\
            (triplet["disease"].lower().strip() == disease.lower().strip())
    return eq

def SYS_PROMPT(general_relations=False, group_relations=False):
    if general_relations:
        options = GENERAL_OPTIONS
    else:
        options = OPTIONS
    if group_relations:
        return f"You are a helpful expert assistant that answers questions about the relations between genes and diseases in a specific input text. You select one or more options from {options}. You only output a comma separated list of the correct options and nothing else."
    return f"You are a helpful assistant that answers questions about the relation between genes and diseases by answering {options} and nothing else."

def _parse_output_labels(output_str):
    """
    Parses output like 'j), o)' -> set of label indices {9, 14}.
    Works for single-label too (e.g., 'l)').
    """
    labels = set()
    for part in output_str.split(","):
        part = part.strip()
        if not part:
            continue
        ch = part[0]
        if "a" <= ch <= "z":
            idx = ord(ch) - ord("a")
            labels.add(idx)
    return labels


def multilabel_stratified_subsample(data, pct, seed, key_fn_multi, min_per_label=0):
    """
    Iterative multi-label stratified subsample.
    Preserves marginal per-label distribution (not per-combination).
    - key_fn_multi(x) -> iterable of label indices for sample x.
    - min_per_label: lower bound per label (0 to disable).
    Deterministic given `seed`.
    """
    if not data or pct <= 0.0:
        return []
    total = len(data)
    n_keep = max(1, int(round(total * pct))) if pct > 0.0 else 0
    if n_keep >= total:
        return data

    rng = random.Random(seed)

    # Build label sets per sample
    y = []
    all_labels = set()
    for x in data:
        labs = set(key_fn_multi(x))
        y.append(labs)
        all_labels.update(labs)

    # Per-label counts and desired targets
    per_label_count = {l: 0 for l in all_labels}
    for labs in y:
        for l in labs:
            per_label_count[l] += 1

    # Desired per-label counts after subsampling
    desired = {}
    for l, c in per_label_count.items():
        want = int(round(c * pct))
        if min_per_label:
            want = max(want, min(min_per_label, c))
        desired[l] = min(want, c)

    # Availability map: label -> set(sample_idx)
    avail = {l: set() for l in all_labels}
    for i, labs in enumerate(y):
        for l in labs:
            avail[l].add(i)

    selected = []
    selected_set = set()

    # Remaining desired per label
    rem = desired.copy()

    # Helper: pick next label to fulfill (prefer rare/under-served labels)
    def pick_label():
        # Choose label with rem > 0; break ties by fewest available remaining candidates
        candidates = [(l, rem[l]) for l in rem if rem[l] > 0]
        if not candidates:
            return None
        # Sort by remaining needed desc, then by availability asc
        candidates.sort(key=lambda kv: (-kv[1], len([i for i in avail[kv[0]] if i not in selected_set])))
        return candidates[0][0]

    while len(selected) < n_keep:
        l = pick_label()
        if l is None:
            break
        # candidates that contain label l and are not yet selected
        cand = [i for i in avail[l] if i not in selected_set]
        if not cand:
            rem[l] = 0
            continue

        # Prefer samples that cover the fewest still‑needed labels to avoid overfitting other labels
        def need_score(i):
            return sum(1 for lab in y[i] if rem.get(lab, 0) > 0)

        # Among minimal need_score, break ties by random choice for reproducibility
        cand.sort(key=lambda i: (need_score(i), rng.random()))
        i_star = cand[0]

        selected.append(i_star)
        selected_set.add(i_star)
        # Decrease remaining needs for all labels present in the chosen sample
        for lab in y[i_star]:
            if rem.get(lab, 0) > 0:
                rem[lab] -= 1

        if all(v <= 0 for v in rem.values()):
            break

    # If still short, fill randomly from remaining pool
    if len(selected) < n_keep:
        remaining = [i for i in range(total) if i not in selected_set]
        rng.shuffle(remaining)
        need = n_keep - len(selected)
        selected.extend(remaining[:need])

    # Shuffle final selection for randomness
    rng.shuffle(selected)
    return [data[i] for i in selected]

class BioTriplexQADataset(Dataset):
    def __init__(self, dataset_config, tokenizer, split_name, max_words=None):

        #self.data = json.load(open(dataset_config.data_path))
        self.relation_types = set()
        self.group_relations = dataset_config.group_relations
        self.bert_mode = getattr(dataset_config, "bert_mode", False)
        self.max_words = max_words
        # label space size (used only in bert_mode for classification)
        if self.group_relations:
            self.label_dim = len(GENERAL_REL_LIST) if dataset_config.general_relations else len(RELATIONS)
        else:
            self.label_dim = len(GENERAL_REL_LIST) if dataset_config.general_relations else len(RELATIONS)

        if split_name == "train":
            with open(dataset_config.data_path + "train_para.txt", "r") as f:
                dataset = f.readlines()
        elif split_name == "val":
            with open(dataset_config.data_path + "val_para.txt", "r") as f:
                dataset = f.readlines()
        elif split_name == "test":
            with open(dataset_config.data_path + "test_para.txt", "r") as f:
                dataset = f.readlines()
        else:
            raise ValueError(f"Invalid split name: {split_name}")
        dataset = [json.loads(line) for line in dataset]
        # dataset is split into sentences I want to treat each sentence as a separate example
        new_dataset = []
        return_neg_relations = (dataset_config.return_neg_relations)
        for sample in dataset:
            for idx, sentence in enumerate(sample["sentences"]):
                # strip whitespace from start and end of sentence while counting the number of whitespaces striped from
                # the start of the sentence
                stripped_sentence = sentence.lstrip()
                num_leading_spaces = len(sentence) - len(stripped_sentence)
                entities = self.correct_entity_char_index(sample["ner"][idx], sample["sentences"], idx,
                                                          num_leading_spaces, stripped_sentence)
                relations = self.correct_relation_char_index(sample["relations"][idx], sample["sentences"], idx,
                                                             num_leading_spaces, stripped_sentence, entities,
                                                             return_neg_relations=return_neg_relations)
                if self.group_relations:
                    relations = self.group_relations_by_gene_disease(relations,
                                                                     general_relations=dataset_config.general_relations)
                for relation in relations:
                    new_sample = {
                        "input": sentence.strip(),
                        "output":
                            triplet_to_answer(relation, general_relations=dataset_config.general_relations)\
                                if not self.group_relations else\
                                grouped_triplet_to_answer(relation, general_relations=dataset_config.general_relations),
                        "doc_key": sample["doc_key"] + f"_sentence_{idx}" + "_" +\
                                   triplet_to_underscore_sep(relation, group_relations=self.group_relations),
                        "entities": entities,
                        "relation": relation
                    }
                    new_dataset.append(new_sample)
        self.data = new_dataset
        # remove data that only contains whitespace
        self.data = [item for item in self.data if item["input"].strip()]
        pct = float(getattr(dataset_config, "train_sample_pct", 1.0))
        if split_name == "train" and pct < 1.0:
            pct = max(0.0, min(1.0, pct))
            seed = int(getattr(dataset_config, "train_sample_seed", 42))
            stratify = bool(getattr(dataset_config, "train_sample_stratify", False))
            min_per_label = int(getattr(dataset_config, "train_sample_min_per_label", 0))
            n_total = len(self.data)

            if pct <= 0.0:
                self.data = []
            elif 0 < pct < 1.0 and n_total > 0:
                if stratify:
                    # Multi‑label aware: preserve marginal label distribution per category
                    self.data = multilabel_stratified_subsample(
                        self.data,
                        pct=pct,
                        seed=seed,
                        key_fn_multi=lambda s: _parse_output_labels(s["output"]),
                        min_per_label=min_per_label,
                    )
                else:
                    rng = random.Random(seed)
                    n_keep = max(1, int(round(n_total * pct)))
                    if 0 <= n_keep < n_total:
                        self.data = rng.sample(self.data, n_keep)
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
        self.general_relations = dataset_config.general_relations
        additional_name = ""
        if dataset_config.general_relations:
            additional_name += "general_"
        if dataset_config.group_relations:
            additional_name += "grouped_"
        gold_path = dataset_config.data_path + f"{split_name}_gold_{additional_name}qa.txt"
        if self.group_relations:
            self.num_of_shots = min(max(dataset_config.num_of_shots, 0), len(GROUPED_FEWSHOTS))
            self.fewshots = GROUPED_FEWSHOTS[:self.num_of_shots]
        else:
            self.num_of_shots = min(max(dataset_config.num_of_shots, 0), len(FEWSHOTS))
            self.fewshots = FEWSHOTS[:self.num_of_shots]
        with open(gold_path, "w") as f:
            for item in self.data:
                f.write(json.dumps(item) + "\n")

    def get_weights(self, positive_factor=1, negative_factor=1):
        # return the weights for each example
        weights = []
        weights_per_relation = {}
        for output in self.num_samples_per_relation:
            if output != NEGATIVE_OPTION:
                weights_per_relation[output] = len(self.data) / self.num_samples_per_relation[output] * positive_factor\
                                               / len(self.num_samples_per_relation)
            else:
                weights_per_relation[output] = len(self.data) / self.num_samples_per_relation[output] * negative_factor\
                                               / len(self.num_samples_per_relation)
        for item in self.data:
            weights.append(weights_per_relation[item["output"]])
        return weights

    def group_relations_by_gene_disease(self, relations, general_relations=False):
        # group different relations between the same gene, disease pairs
        grouped_relations = []
        for relation in relations:
            gene = relation["gene"]
            disease = relation["disease"]
            if general_relations:
                relation_relation = None
                for key in GENERAL_REL:
                    if relation["relation"].lower() in GENERAL_REL[key]:
                        relation_relation = key
                        break
                assert relation_relation is not None
            else:
                relation_relation = relation["relation"]
            found_group = False
            for grouped_relation in grouped_relations:
                if triplet_has_equal_gene_and_disease(grouped_relation, gene, disease):
                    grouped_relation["relation"].add(relation_relation)
                    found_group = True
                    break
            if not found_group:
                grouped_relations.append({"gene": gene, "disease": disease, "relation": {relation_relation}})
        for grouped_relation in grouped_relations:
            grouped_relation["relation"] = list(grouped_relation["relation"])
            # sort the relations in alphabetical order
            grouped_relation["relation"].sort()
        return grouped_relations

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
        corrected_entities = BioTriplexQADataset.correct_overlap(corrected_entities, stripped_sentence)
        # remove any trailing whitespace
        corrected_entities = BioTriplexQADataset.remove_trailing_whitespace(corrected_entities, stripped_sentence,
                                                                          sentence_idx, num_leading_spaces, offset, sentences)
        return corrected_entities

    def correct_relation_char_index(self, relations, sentences, sentence_idx, num_leading_spaces, stripped_sentence,
                                    entities, return_neg_relations=False):
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
        corrected_relations = BioTriplexQADataset.relation_remove_trailing_whitespace(corrected_relations, stripped_sentence,
                                                                            sentence_idx, num_leading_spaces, offset, sentences)
        if return_neg_relations:
            # for all the entities that are not in a relation with another entity of the same type, add a negative relation
            # with the entity
            for gene in entities:
                if gene[2] != "GENE":
                    continue
                for disease in entities:
                    if disease[2] != "DISEASE":
                        continue
                    gene_start, gene_end = gene[:2]
                    disease_start, disease_end = disease[:2]
                    found_relation = False
                    for relation in corrected_relations:
                        if link_is_equal(relation, gene, disease, stripped_sentence):
                            found_relation = True
                            break
                    if not found_relation:
                        corrected_relations.append([gene_start, gene_end, disease_start, disease_end, "No Relation"])

        # map genes and disease ids to their respective entities
        corrected_relations = [{"gene": stripped_sentence[relation[0]: relation[1]],
                                "disease": stripped_sentence[relation[2]: relation[3]],
                                "relation": relation[4]} for relation in corrected_relations]
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

    def openai_get_all_input_prompts(self, skip_system=False):
        prompts = {}
        for item in self.data:
            prompt = self.openai_input_to_prompt(item["input"], item["relation"], skip_system=skip_system)
            prompts[item["doc_key"]] = prompt
        if skip_system:
            sys_prompt = SYS_PROMPT(general_relations=self.general_relations, group_relations=self.group_relations)
            return prompts, sys_prompt
        return prompts

    def openai_input_to_prompt(self, input_text, triplet, skip_system=False):
        instr = INSTRUCTION(triplet, general_relations=self.general_relations, group_relations=self.group_relations)
        sys_prompt = SYS_PROMPT(general_relations=self.general_relations, group_relations=self.group_relations)

        messages = []

        # Add system message
        if not skip_system:
            messages.append({"role": "system", "content": sys_prompt})

        # Add few-shot examples
        for fewshot in self.fewshots:
            fewshot_instruction = fewshot_to_instruction(fewshot, general_relations=self.general_relations,
                                                         group_relations=self.group_relations)
            user_msg = f"### Instruction:\n{fewshot_instruction}\n### Input:\n{fewshot['input']}"
            assistant_msg = f"### Response:\n{fewshot['output_gen']}"\
                if self.general_relations else f"### Response:\n{fewshot['output']}"
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})

        # Add final user query
        user_query = f"### Instruction:\n{instr}\n### Input:\n{input_text}"
        messages.append({"role": "user", "content": user_query})

        return messages


    def input_to_prompt(self, input_text, triplet):
        # prompt = f"### Instruction:\n{INSTRUCTION}\n\n### Input:\n{input_text}\n\n### Response:\n"
        instr = INSTRUCTION(triplet, general_relations=self.general_relations, group_relations=self.group_relations)
        sys_prompt = SYS_PROMPT(general_relations=self.general_relations, group_relations=self.group_relations)
        fewshot_prompts = ""
        for fewshot in self.fewshots:
            fewshot_prompts += "<|start_header_id|>user<|end_header_id|>"
            fewshot_prompts += f"### Instruction:\n{fewshot_to_instruction(fewshot, general_relations=self.general_relations, group_relations=self.group_relations)}"
            fewshot_prompts += f"\n### Input:\n{fewshot['input']}"
            fewshot_prompts += "\n\n" + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
            if self.general_relations:
                fewshot_prompts += f"### Response:\n{fewshot['output_gen']}"
            else:
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

    def _output_to_single_label(self, output_str):
        # output like "l)" -> take first character
        ch = output_str.strip()[0]
        return ord(ch) - ord('a')

    def _output_to_multi_label(self, output_str):
        # output like "j), o)" -> multi-hot vector
        vec = [0] * self.label_dim
        parts = [p.strip() for p in output_str.split(",")]
        for p in parts:
            if not p:
                continue
            ch = p[0]
            idx = ord(ch) - ord('a')
            if 0 <= idx < self.label_dim:
                vec[idx] = 1
        return vec

    def _build_bert_item(self, item):
        """
        Builds a BERT classification sample.
        Modified: use a minimal question (genes + diseases only) to reduce prompt length
        and allow more room for the context, minimizing truncation.
        """
        # item["relation"] is the triplet dict (gene, disease, relation or relations)
        triplet = item["relation"]
        gene = triplet["gene"]
        disease = triplet["disease"]

        # Minimal question segment (Segment A)
        # Keep very short to preserve context tokens
        question_text = f"Gene: {gene} Disease: {disease}"

        # Context (Segment B)
        context_text = item["input"]

        # Tokenize without specials
        q_ids = self.tokenizer.encode(question_text, add_special_tokens=False)
        c_ids = self.tokenizer.encode(context_text, add_special_tokens=False)

        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        if cls_id is None or sep_id is None:
            raise ValueError("Tokenizer must have CLS and SEP tokens for BERT mode.")

        # Truncate ONLY context if needed (question kept intact)
        if self.max_words is not None:
            # Reserve: 1 [CLS] + len(q_ids) + 1 [SEP] + context + 1 [SEP]
            overhead = 1 + len(q_ids) + 1 + 1  # CLS + q + SEP + final SEP
            max_ctx = self.max_words - overhead
            if max_ctx < 0:
                # Extreme edge: fall back to trimming question as last resort
                q_trim = self.max_words - 3
                q_ids = q_ids[:max(0, q_trim)]
                c_ids = []
            else:
                if len(c_ids) > max_ctx:
                    c_ids = c_ids[:max_ctx]

        input_ids = [cls_id] + q_ids + [sep_id] + c_ids + [sep_id]

        # token_type_ids: 0 for question segment (CLS + question + SEP), 1 for context + final SEP
        token_type_ids = [0] * (1 + len(q_ids) + 1) + [1] * (len(c_ids) + 1)
        attention_mask = [1] * len(input_ids)

        # Labels
        if self.group_relations:
            # Multi-label (list of relation codes)
            if self.general_relations:
                # When general_relations: relation list already generalized upstream
                rel_list = triplet["relation"] if isinstance(triplet["relation"], list) else [triplet["relation"]]
                # Map each to its index in GENERAL_REL_LIST
                label_vec = [0] * self.label_dim
                for r in rel_list:
                    idx = GENERAL_REL_LIST.index(r)
                    label_vec[idx] = 1
                labels = label_vec
            else:
                rel_list = triplet["relation"] if isinstance(triplet["relation"], list) else [triplet["relation"]]
                label_vec = [0] * self.label_dim
                for r in rel_list:
                    idx = RELATIONS.index(r.lower())
                    label_vec[idx] = 1
                labels = label_vec
        else:
            # Single-label
            if self.general_relations:
                idx_rel = GENERAL_REL_LIST.index(triplet["relation"])
            else:
                idx_rel = RELATIONS.index(triplet["relation"].lower())
            labels = idx_rel

        # Sample weight (retain existing balancing logic if enabled)
        if self.upweight_minority_class:
            # If weights map prepared elsewhere (num_samples_per_relation), use inverse freq
            if not hasattr(self, "_class_weights_cached"):
                # Build simple inverse frequency weights
                counts = {}
                for s in self.data:
                    if self.group_relations:
                        rels = s["relation"]["relation"] if isinstance(s["relation"]["relation"], list) else [s["relation"]["relation"]]
                        for r in rels:
                            counts[r] = counts.get(r, 0) + 1
                    else:
                        r = s["relation"]["relation"]
                        counts[r] = counts.get(r, 0) + 1
                max_c = max(counts.values()) if counts else 1
                self._class_weights_cached = {k: max_c / v for k, v in counts.items()}
            if self.group_relations:
                rels = triplet["relation"] if isinstance(triplet["relation"], list) else [triplet["relation"]]
                # Average weights for multi-label
                sw = sum(self._class_weights_cached.get(r if self.general_relations else r.lower(), 1.0) for r in rels) / max(1, len(rels))
            else:
                key = triplet["relation"] if self.general_relations else triplet["relation"].lower()
                sw = self._class_weights_cached.get(key, 1.0)
            sample_weight = sw
        else:
            sample_weight = 1.0

        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "weight": sample_weight,
        }


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # self.input_seen.add(index)

        item = self.data[index]
        debug = False #"4636297_Discussion_sentence_0" #"3979708_Introduction_sentence_0"  # (item["doc_key"] == "9818593_3. Results_sentence_0")
        # New BERT path (does not alter preprocessing or sample selection)
        if self.bert_mode:
            return self._build_bert_item(item)

        # Original generative path (unchanged)
        IGNORE_INDEX = -100
        # prompt = item['input']#f"item['input']\n\n"
        prompt_prefix, prompt_input, prompt_suffix = self.input_to_prompt(item["input"], item["relation"])
        prompt = prompt_prefix + prompt_input + prompt_suffix
        # example = prompt + item["output"]
        example = prompt + "\n### Response:\n" + item["output"]
        if self.upweight_minority_class:
            weight = POSITIVE_WEIGHT if item["output"] != NEGATIVE_OPTION else NEGATIVE_WEIGHT
        else:
            weight = 1.0
        example_ids = self.tokenizer.encode(example)
        example_ids.append(self.tokenizer.eos_token_id)
        example = torch.tensor(example_ids, dtype=torch.int64)
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
            assert len(prompt_prefix) + len(prompt_input) + len(prompt_suffix) == len(self.tokenizer.encode(prompt))
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
            prompt_ids = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.int64)
            labels[:len(prompt_ids)] = -1

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
    for mode in "test", "train", "val":
        dataset = BioTriplexQADataset(dataset_config, tokenizer, mode, max_words=None,)
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

# python
import os
import json
import math
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from tqdm.auto import tqdm
from .biotriplex_qakshot_dataset import BioTriplexQADataset, RELATIONS, GENERAL_REL_LIST

try:
    from torch.amp import autocast as _autocast, GradScaler as _GradScaler
    _USE_NEW_AMP = True
except (ImportError, AttributeError):
    from torch.cuda.amp import autocast as _autocast, GradScaler as _GradScaler
    _USE_NEW_AMP = False

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


@dataclass
class BioTriplexConfig:
    dataset: str = "biotriplex_qa_bert_dataset"
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
    bert_mode: bool = True
    train_sample_pct: float = 1.0
    train_sample_seed: int = 42
    train_sample_stratify: bool = False
    train_sample_min_per_label: int = 0


def build_multilabel_sampler(ds: Dataset, seed: int) -> WeightedRandomSampler:
    """
    Create a WeightedRandomSampler for multilabel data.
    Each sample weight = average inverse frequency of its labels.
    Works with BioTriplexQADataset in bert mode (uses ds.data to avoid tokenization).
    """
    # Count per label and gather per‑sample label indices
    num_labels = ds.label_dim
    counts = [0] * num_labels
    per_sample_labels = []

    for s in ds.data:
        # s["relation"] is {"gene", "disease", "relation": list|str}
        rels = s["relation"]["relation"]
        rel_list = rels if isinstance(rels, list) else [rels]
        idxs = []
        if ds.general_relations:
            for r in rel_list:
                idxs.append(GENERAL_REL_LIST.index(r))
        else:
            for r in rel_list:
                idxs.append(RELATIONS.index(r.lower()))
        # dedupe labels per sample to not overcount
        idxs = sorted(set(idxs))
        per_sample_labels.append(idxs)
        for i in idxs:
            counts[i] += 1

    # Inverse frequency
    inv = [1.0 / max(1, c) for c in counts]

    # Sample weights = mean of inverse frequencies of present labels
    weights = []
    for labs in per_sample_labels:
        if not labs:
            weights.append(1.0)
        else:
            w = sum(inv[i] for i in labs) / len(labs)
            weights.append(w)

    # Optional normalization (keeps average ~1.0)
    s = sum(weights)
    if s > 0:
        scale = len(weights) / s
        weights = [w * scale for w in weights]

    gen = torch.Generator()
    gen.manual_seed(seed)
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True, generator=gen)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="dmis-lab/biobert-v1.1")
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="outputs/biotriplex_bert")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--gradient_accumulation", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--general_relations", action="store_true")
    p.add_argument("--group_relations", action="store_true")
    p.add_argument("--no_neg", action="store_true")
    p.add_argument("--upweight", action="store_true")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--save_every_epoch", action="store_true")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="biotriplex-bert")
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_tags", type=str, default="")
    p.add_argument("--auto_shift_labels", action="store_true",
                   help="Auto shift labels to start at 0 if min label > 0 (single-label mode).")
    p.add_argument("--debug_samples", type=int, default=1,
                   help="Number of samples to show each epoch from first train & val batch.")
    # Threshold optimization (multi-label)
    p.add_argument("--optimize_threshold", action="store_true",
                   help="Search best decision threshold on validation set (multi-label only).")
    p.add_argument("--threshold_search_min", type=float, default=0.05)
    p.add_argument("--threshold_search_max", type=float, default=0.95)
    p.add_argument("--threshold_search_step", type=float, default=0.05)
    p.add_argument("--threshold_metric", type=str, default="micro_f1",
                   help="Metric key to maximize when searching threshold.")
    p.add_argument("--train_sample_pct", type=float, default=1.0)
    p.add_argument("--train_sample_stratify", action="store_true")
    p.add_argument("--train_sample_min_per_label", type=int, default=0)
    p.add_argument("--train_sample_seed", type=int, default=42)
    p.add_argument("--no_amp", action="store_true", help="Disable mixed precision for determinism.")
    p.add_argument("--upsample", action="store_true",
                   help="Enable WeightedRandomSampler upsampling for multilabel train set.")
    return p.parse_args()


def make_deterministic(seed: int):
    import os, random, numpy as np, torch
    # For deterministic matmuls in cuBLAS (set before first CUDA use)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    # Global RNGs
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # cuDNN/TF32 and deterministic kernels
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Enforce deterministic algorithms where possible
    torch.use_deterministic_algorithms(True, warn_only=True)


# def set_seed(seed: int):
#     import random, numpy as np
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)


class BertCollator:
    def __init__(self, pad_token_id: int, max_length: int):
        self.pad_token_id = pad_token_id
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        multi_label = isinstance(batch[0]["labels"], list) and not isinstance(batch[0]["labels"], (bytes, str))
        trimmed = []
        for ex in batch:
            if len(ex["input_ids"]) > self.max_length:
                ex = ex.copy()
                ex["input_ids"] = ex["input_ids"][:self.max_length]
                ex["token_type_ids"] = ex["token_type_ids"][:self.max_length]
                ex["attention_mask"] = ex["attention_mask"][:self.max_length]
            trimmed.append(ex)
        batch = trimmed

        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids, token_type_ids, attn, labels, weights = [], [], [], [], []
        for ex in batch:
            l = len(ex["input_ids"])
            pad = max_len - l
            input_ids.append(ex["input_ids"] + [self.pad_token_id] * pad)
            token_type_ids.append(ex["token_type_ids"] + [0] * pad)
            attn.append(ex["attention_mask"] + [0] * pad)
            weights.append(ex.get("weight", 1.0))
            labels.append(ex["labels"])

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
        attention_mask = torch.tensor(attn, dtype=torch.long)
        sample_weight = torch.tensor(weights, dtype=torch.float)
        if multi_label:
            labels = torch.tensor(labels, dtype=torch.float)
        else:
            labels = torch.tensor(labels, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "sample_weight": sample_weight,
        }

def build_dataset(split: str, args, tokenizer):
    # Train: honor --no_neg (treat as augmentation)
    # Val/Test: always include negative relations for unbiased eval
    if split == "train":
        return_neg_relations = not args.no_neg
        train_sample_pct = args.train_sample_pct
        train_sample_stratify = args.train_sample_stratify
        train_sample_min_per_label = args.train_sample_min_per_label
    else:
        return_neg_relations = False
        train_sample_pct = 1.0
        train_sample_stratify = False
        train_sample_min_per_label = 0

    cfg = BioTriplexConfig(
        data_path=args.data_path,
        general_relations=args.general_relations,
        group_relations=args.group_relations,
        upweight_minority_class=args.upweight,
        return_neg_relations=return_neg_relations,
        bert_mode=True,
        train_sample_pct=train_sample_pct,
        train_sample_seed=args.train_sample_seed,
        train_sample_stratify=train_sample_stratify,
        train_sample_min_per_label=train_sample_min_per_label,
    )
    return BioTriplexQADataset(cfg, tokenizer, split_name=split, max_words=args.max_length)


def move_to(batch, device):
    return {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}


def init_wandb(args, num_labels):
    if not args.wandb or not _WANDB_AVAILABLE:
        return False
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        entity=args.wandb_entity,
        tags=[t for t in args.wandb_tags.split(",") if t],
        config=vars(args) | {"num_labels": num_labels},
    )
    return True


def _sanitize_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)


def extract_class_names(ds, num_labels: int) -> List[str]:
    candidates = []
    for attr in ["id2label", "class_names", "label_names", "relation_labels", "labels_list"]:
        if hasattr(ds, attr):
            val = getattr(ds, attr)
            if isinstance(val, dict):
                try:
                    ordered = [val[i] for i in sorted(val.keys())]
                except Exception:
                    ordered = list(val.values())
                candidates = ordered
                break
            if isinstance(val, list):
                candidates = val
                break
    if not candidates or len(candidates) != num_labels:
        candidates = [f"class_{i}" for i in range(num_labels)]
    out = []
    seen = set()
    for c in candidates:
        s = _sanitize_name(str(c))
        if s in seen:
            base = s
            k = 1
            while f"{base}_{k}" in seen:
                k += 1
            s = f"{base}_{k}"
        seen.add(s)
        out.append(s)
    return out


def metrics_single(preds: torch.Tensor, labels: torch.Tensor, class_names: Optional[List[str]] = None) -> Dict[str, float]:
    num_classes = int(labels.max().item()) + 1
    if not class_names or len(class_names) != num_classes:
        class_names = [f"class_{i}" for i in range(num_classes)]
    tp = fp = fn = 0
    for c in range(num_classes):
        p = preds == c
        t = labels == c
        tp += (p & t).sum().item()
        fp += (p & (~t)).sum().item()
        fn += ((~p) & t).sum().item()
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    out = dict(
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=(preds == labels).float().mean().item(),
        micro_precision=precision,
        micro_recall=recall,
        micro_f1=f1,
    )
    for c in range(num_classes):
        p = preds == c
        t = labels == c
        c_tp = (p & t).sum().item()
        c_fp = (p & (~t)).sum().item()
        c_fn = ((~p) & t).sum().item()
        c_prec = c_tp / (c_tp + c_fp + 1e-12)
        c_rec = c_tp / (c_tp + c_fn + 1e-12)
        c_f1 = 2 * c_prec * c_rec / (c_prec + c_rec + 1e-12)
        name = class_names[c]
        out[f"{name}_precision"] = c_prec
        out[f"{name}_recall"] = c_rec
        out[f"{name}_f1"] = c_f1
    return out


def metrics_multi(logits: torch.Tensor, labels: torch.Tensor, thr: float,
                  class_names: Optional[List[str]] = None) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= thr).float()
    tp = (preds * labels).sum().item()
    fp = (preds * (1 - labels)).sum().item()
    fn = ((1 - preds) * labels).sum().item()
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    out = dict(
        precision=precision,
        recall=recall,
        f1=f1,
        threshold=thr,
        micro_precision=precision,
        micro_recall=recall,
        micro_f1=f1,
    )
    num_classes = labels.size(1)
    if not class_names or len(class_names) != num_classes:
        class_names = [f"class_{i}" for i in range(num_classes)]
    for c in range(num_classes):
        p = preds[:, c]
        t = labels[:, c]
        c_tp = (p * t).sum().item()
        c_fp = (p * (1 - t)).sum().item()
        c_fn = ((1 - p) * t).sum().item()
        c_prec = c_tp / (c_tp + c_fp + 1e-12)
        c_rec = c_tp / (c_tp + c_fn + 1e-12)
        c_f1 = 2 * c_prec * c_rec / (c_prec + c_rec + 1e-12)
        name = class_names[c]
        out[f"{name}_precision"] = c_prec
        out[f"{name}_recall"] = c_rec
        out[f"{name}_f1"] = c_f1
    return out


def decode_debug(tokenizer, input_ids):
    return tokenizer.decode(input_ids, skip_special_tokens=True)[:]


def debug_batch(model, tokenizer, batch, device, multi, tag: str, max_samples: int):
    model.eval()
    with torch.no_grad():
        b = {k: v[:max_samples].to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        out = model(
            input_ids=b["input_ids"],
            attention_mask=b["attention_mask"],
            token_type_ids=b["token_type_ids"],
        )
        logits = out.logits.cpu()
        if multi:
            probs = torch.sigmoid(logits)
        else:
            probs = torch.softmax(logits, dim=-1)
    for i in range(min(max_samples, b["input_ids"].size(0))):
        text = decode_debug(tokenizer, b["input_ids"][i].cpu().tolist())
        target = b["labels"][i].cpu().tolist()
        pred_vec = probs[i].tolist()
        if not multi:
            pred_class = int(torch.argmax(probs[i]).item())
            print(f"[{tag}] Sample {i} target={target} pred_class={pred_class} probs={pred_vec}")
        else:
            pred_bin = (probs[i] >= 0.5).int().tolist()
            print(f"[{tag}] Sample {i} target_multi={target} pred_bin={pred_bin} probs={pred_vec}")
        print(f"[{tag}] Text: {text}")


@torch.no_grad()
def evaluate(model, dataloader, device, multi: bool, threshold: float, desc: str,
             tokenizer=None, debug_samples=0, compute_f1: bool = True,
             class_names: Optional[List[str]] = None, return_logits: bool = False):
    model.eval()
    all_logits, all_labels = [], []
    first_batch = None
    total_loss_sum = 0.0
    total_samples = 0

    bce_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
    ce_loss = torch.nn.CrossEntropyLoss(reduction="none")

    for batch in tqdm(dataloader, desc=desc, leave=False):
        if first_batch is None:
            first_batch = batch
        batch_dev = move_to(batch, device)
        with torch.no_grad():
            out = model(
                input_ids=batch_dev["input_ids"],
                attention_mask=batch_dev["attention_mask"],
                token_type_ids=batch_dev["token_type_ids"],
            )
            logits = out.logits
            if multi:
                target = batch_dev["labels"].float()
                loss_raw = bce_loss(logits, target)
                sw = batch_dev["sample_weight"].unsqueeze(1)
                per_sample_loss = (loss_raw * sw).mean(dim=1)
            else:
                target = batch_dev["labels"].long()
                ce = ce_loss(logits, target)
                sw = batch_dev["sample_weight"]
                per_sample_loss = ce * sw
            total_loss_sum += per_sample_loss.sum().item()
            total_samples += per_sample_loss.size(0)

        all_logits.append(logits.cpu())
        all_labels.append(batch["labels"].cpu())

    assert total_samples == len(dataloader.dataset), (
        f"Sample count mismatch: counted={total_samples} dataset={len(dataloader.dataset)}"
    )

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)

    if multi:
        base_metrics = metrics_multi(logits, labels, threshold, class_names=class_names)
        metrics = base_metrics
    else:
        preds = logits.argmax(dim=-1)
        base_metrics = metrics_single(preds, labels, class_names=class_names)
        metrics = base_metrics
        if desc.lower().startswith("validation"):
            u_lab, c_lab = labels.unique(return_counts=True)
            u_pred, c_pred = preds.unique(return_counts=True)
            print("Validation label distribution:", dict(zip(u_lab.tolist(), c_lab.tolist())))
            print("Validation pred distribution:", dict(zip(u_pred.tolist(), c_pred.tolist())))

    # Added support counts
    num_classes = len(class_names) if class_names else (labels.size(1) if multi else int(labels.max().item()) + 1)
    if multi:
        supports = labels.sum(dim=0)  # number of positives per class
        metrics["total_samples"] = int(labels.size(0))
        metrics["total_positive_labels"] = float(labels.sum().item())
    else:
        supports = torch.tensor([(labels == i).sum() for i in range(num_classes)])
        metrics["total_samples"] = int(labels.size(0))
    for i in range(num_classes):
        metrics[f"{class_names[i]}_support"] = float(supports[i].item())

    avg_loss = total_loss_sum / max(1, total_samples)
    metrics["loss"] = avg_loss

    if debug_samples > 0 and tokenizer is not None and first_batch is not None:
        debug_batch(model, tokenizer, first_batch, device, multi, f"{desc}Debug", debug_samples)

    if return_logits:
        return metrics, logits, labels
    return metrics


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    # Save initial run config
    config_path = os.path.join(args.output_dir, "run_config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)
    print(f"Saved run config to {config_path}")
    make_deterministic(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    train_ds = build_dataset("train", args, tokenizer)
    val_ds = build_dataset("val", args, tokenizer)
    test_ds = build_dataset("test", args, tokenizer)

    num_labels = train_ds.label_dim
    multi = args.group_relations

    class_names = extract_class_names(train_ds, num_labels)

    if not multi:
        sample_labels = []
        for i in range(min(200, len(train_ds))):
            sample_labels.append(train_ds[i]["labels"])
        if sample_labels:
            lmin = min(sample_labels)
            if lmin > 0:
                print(f"Observed train label min={lmin} max={max(sample_labels)}")
                print("WARNING: Labels start above 0. Enable --auto_shift_labels to shift.")
            if args.auto_shift_labels and lmin > 0:
                shift = lmin
                class ShiftedDataset(Dataset):
                    def __init__(self, base, shift_):
                        self.base = base; self.shift = shift_
                    def __len__(self): return len(self.base)
                    def __getitem__(self, idx):
                        ex = self.base[idx].copy()
                        ex["labels"] = ex["labels"] - self.shift
                        return ex
                train_ds = ShiftedDataset(train_ds, shift)
                val_ds = ShiftedDataset(val_ds, shift)
                test_ds = ShiftedDataset(test_ds, shift)
                num_labels = max(sample_labels) - shift + 1
                print(f"Shifted labels by {shift}. New num_labels={num_labels}")
                class_names = [f"class_{i}" for i in range(num_labels)]

    collator = BertCollator(pad_token_id=tokenizer.pad_token_id, max_length=args.max_length)

    # Use sampler only for training when enabled and in multilabel mode
    if args.upsample and multi:
        print("Using WeightedRandomSampler for multilabel upsampling on train set.")
        train_sampler = build_multilabel_sampler(train_ds, seed=args.seed)
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, collate_fn=collator)
    else:
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    config = AutoConfig.from_pretrained(args.model_name, num_labels=num_labels)
    if multi:
        config.problem_type = "multi_label_classification"
    config.id2label = {i: n for i, n in enumerate(class_names)}
    config.label2id = {n: i for i, n in enumerate(class_names)}

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, config=config).to(device)

    use_wandb = init_wandb(args, num_labels)
    if use_wandb:
        wandb.watch(model, log="all", log_freq=200)

    total_steps = math.ceil(len(train_dl) / args.gradient_accumulation) * args.epochs
    warmup = int(total_steps * args.warmup_ratio)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)
    scaler = _GradScaler(enabled=torch.cuda.is_available())

    best_val_loss = float("inf")
    best_dir = os.path.join(args.output_dir, "best")
    global_step = 0

    amp_enabled = torch.cuda.is_available() and not args.no_amp

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        optimizer.zero_grad()
        pbar = tqdm(train_dl, desc=f"Epoch {epoch} Training", leave=False)
        first_train_batch = None
        for step, batch in enumerate(pbar):
            if first_train_batch is None:
                first_train_batch = batch
            batch_dev = move_to(batch, device)
            if _USE_NEW_AMP:
                ac_ctx = _autocast(device_type="cuda", enabled=amp_enabled)
            else:
                ac_ctx = _autocast(enabled=amp_enabled)
            with ac_ctx:
                outputs = model(
                    input_ids=batch_dev["input_ids"],
                    attention_mask=batch_dev["attention_mask"],
                    token_type_ids=batch_dev["token_type_ids"],
                    labels=None if multi else batch_dev["labels"],
                )
                logits = outputs.logits
                if multi:
                    target = batch_dev["labels"].float()
                    loss_raw = torch.nn.BCEWithLogitsLoss(reduction="none")(logits, target)
                    sw = batch_dev["sample_weight"].unsqueeze(1)
                    loss = (loss_raw * sw).mean()
                else:
                    target = batch_dev["labels"].long()
                    ce = torch.nn.CrossEntropyLoss(reduction="none")(logits, target)
                    sw = batch_dev["sample_weight"]
                    loss = (ce * sw).mean()
            scaler.scale(loss / args.gradient_accumulation).backward()
            running += loss.item()
            if (step + 1) % args.gradient_accumulation == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                if use_wandb:
                    wandb.log({
                        "train/loss_step": loss.item(),
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/epoch_progress": epoch + step / len(train_dl),
                        "train/global_step": global_step
                    }, step=global_step)
            pbar.set_postfix(loss=loss.item())

        if first_train_batch is not None and args.debug_samples > 0:
            print(f"--- Epoch {epoch} Train Debug ---")
            debug_batch(model, tokenizer, first_train_batch, device, multi, "TrainDebug", args.debug_samples)

        val_metrics = evaluate(
            model, val_dl, device, multi, args.threshold,
            desc="Validation", tokenizer=tokenizer,
            debug_samples=args.debug_samples, compute_f1=False,
            class_names=class_names
        )
        val_loss = val_metrics["loss"]
        ep_loss = running / len(train_dl)
        prec = val_metrics.get("precision", 0.0)
        rec = val_metrics.get("recall", 0.0)
        print(f"Epoch {epoch} | TrainLoss {ep_loss:.4f} | ValLoss {val_loss:.4f} | ValP {prec:.4f} ValR {rec:.4f}")

        if use_wandb:
            wandb.log({
                "val/loss": val_loss,
                "val/precision": prec,
                "val/recall": rec,
                "train/epoch_loss": ep_loss,
                "epoch": epoch
            }, step=global_step)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(best_dir, exist_ok=True)
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            with open(os.path.join(best_dir, "val_metrics.txt"), "w") as f:
                f.write(str(val_metrics))
            print(f"Saved best model to {best_dir}")
            if use_wandb:
                wandb.log({"val/best_loss": best_val_loss}, step=global_step)

        if args.save_every_epoch:
            ep_dir = os.path.join(args.output_dir, f"epoch_{epoch}")
            os.makedirs(ep_dir, exist_ok=True)
            model.save_pretrained(ep_dir)
            tokenizer.save_pretrained(ep_dir)

    print("Loading best model...")
    best_model = AutoModelForSequenceClassification.from_pretrained(best_dir).to(device)

    used_threshold = args.threshold
    if multi and args.optimize_threshold:
        print("Optimizing threshold on validation set...")
        val_full_metrics, val_logits, val_labels = evaluate(
            best_model, val_dl, device, multi, args.threshold,
            desc="Validation-Final", tokenizer=tokenizer,
            debug_samples=0, compute_f1=True,
            class_names=class_names, return_logits=True
        )
        best_thr = args.threshold
        best_score = -1.0
        thr = args.threshold_search_min
        while thr <= args.threshold_search_max + 1e-9:
            m = metrics_multi(val_logits, val_labels, thr, class_names=class_names)
            score = m.get(args.threshold_metric, m.get("micro_f1", 0.0))
            if score > best_score:
                best_score = score
                best_thr = thr
            thr += args.threshold_search_step
        used_threshold = best_thr
        print(f"Selected threshold={best_thr:.4f} maximizing {args.threshold_metric}={best_score:.4f}")
        with open(os.path.join(best_dir, "selected_threshold.txt"), "w") as f:
            f.write(f"{best_thr}\nmetric={args.threshold_metric}\nscore={best_score}\n")

    test_metrics = evaluate(
        best_model, test_dl, device, multi, used_threshold,
        desc="Test", tokenizer=tokenizer,
        debug_samples=args.debug_samples, compute_f1=True,
        class_names=class_names
    )

    # Print support counts (added)
    print("Test per-class support counts:")
    for cname in class_names:
        key = f"{cname}_support"
        if key in test_metrics:
            print(f"{cname}: {int(test_metrics[key])}")

    # JSON serialization unchanged, now includes *_support keys automatically
    json_test_metrics = {
        k: (float(v) if isinstance(v, (int, float)) else v)
        for k, v in test_metrics.items()
    }
    test_metrics_path = os.path.join(args.output_dir, "test_metrics.json")
    with open(test_metrics_path, "w") as f:
        json.dump({"threshold": used_threshold, **json_test_metrics}, f, indent=2, sort_keys=True)
    print(f"Saved test metrics to {test_metrics_path}")
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "config": vars(args),
            "used_threshold": used_threshold,
            "test_metrics": json_test_metrics
        }, f, indent=2, sort_keys=True)
    print(f"Test (threshold={used_threshold:.4f}) metrics:")
    for k, v in test_metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
    with open(os.path.join(best_dir, "test_metrics.txt"), "w") as f:
        f.write("threshold=" + str(used_threshold) + "\n")
        f.write(str(test_metrics))
    if use_wandb:
        wandb.log({
            "threshold": used_threshold,
            **{f"test/{k}": v for k, v in test_metrics.items() if isinstance(v, (float, int))}
        }, step=global_step)


if __name__ == "__main__":
    main()

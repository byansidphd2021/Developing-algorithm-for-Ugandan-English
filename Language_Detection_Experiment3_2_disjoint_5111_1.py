import os
import re
import random
import unicodedata
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
import torch

from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    roc_curve,
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed,
    AutoConfig,
    EarlyStoppingCallback,
    default_data_collator,
)

import transformers, inspect


# =========================
# 0) Config
# =========================

SEED = 42
set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

TRAIN_DATA_PATH = "Training_Dataset_cleaned_edited_sorted_by_UGE_content51.csv"
TEST_DATA_PATH  = "Test_Dataset_new_UGE_content3.csv"

OUTPUT_BASE_DIR = "UGE_Experiments_Outputs_32_2_365_Disjoint__LeakageSafe"

MODEL_NAMES = {
    "Serengeti":"UBC-NLP/serengeti-E250",
    #"afro_xlm_r_large": "Davlan/afro-xlmr-large",
    #"afro_xlm_r_base": "Davlan/afro-xlmr-base",
    #"afro_xlm_r_small": "Davlan/afro-xlmr-small",
    #"afro_xlm_r_mini": "Davlan/afro-xlmr-mini",
    #"mbert": "bert-base-multilingual-cased",
    #"afrimbart": "masakhane/afrimbart_fr_bam_news",
    #"XLM-R-miniLM":"microsoft/Multilingual-MiniLM-L12-H384",
    #"XLM-R-base":"FacebookAI/xlm-roberta-base",
    #"XLM-R-Lagre":"FacebookAI/xlm-roberta-large"
}

UGE_TAG_PATTERN = re.compile(r"<\s*UGE\s*>(.*?)<\s*/\s*UGE\s*>", re.IGNORECASE | re.DOTALL)

MAX_LENGTH = 256
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
LOGGING_STEPS = 50

SAVE_STRATEGY = "no"

NUM_SHARDS = int(os.environ.get("NUM_SHARDS", "1"))
SHARD_ID = int(os.environ.get("SHARD_ID", "0"))
MERGE_ONLY = os.environ.get("MERGE_ONLY", "0") == "1"

# =========================
# Disjointness / Leakage controls
# =========================
DISJOINT_MODE = os.environ.get("DISJOINT_MODE", "cue").strip().lower()
DISJOINT_UNIT = os.environ.get("DISJOINT_UNIT", "span").strip().lower()
DISJOINT_APPROXIMATE = os.environ.get("DISJOINT_APPROXIMATE", "0") == "1"
DISJOINT_JACCARD_THRESHOLD = float(os.environ.get("DISJOINT_JACCARD_THRESHOLD", "0.6"))

DISJOINT_EMBEDDING_THRESHOLD = float(os.environ.get("DISJOINT_EMBEDDING_THRESHOLD", "0.85"))
DISJOINT_EMBEDDING_BATCH_SIZE = int(os.environ.get("DISJOINT_EMBEDDING_BATCH_SIZE", "32"))
DISJOINT_EMBEDDING_MODEL = os.environ.get("DISJOINT_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

DISJOINT_TEST_ACTION = os.environ.get("DISJOINT_TEST_ACTION", "drop").strip().lower()

FAIL_IF_TRAINVAL_TEST_CUE_OVERLAP = os.environ.get("FAIL_IF_TRAINVAL_TEST_CUE_OVERLAP", "1") == "1"
DROP_EXACT_DUPLICATES_ACROSS_SPLITS = os.environ.get("DROP_EXACT_DUPLICATES_ACROSS_SPLITS", "1") == "1"

REPORT_FIXED_THRESHOLD_0_5 = os.environ.get("REPORT_FIXED_THRESHOLD_0_5", "1") == "1"
THRESHOLD_TUNING_ON_VAL = os.environ.get("THRESHOLD_TUNING_ON_VAL", "1") == "1"

sent_orig_col = "sentence"
lab_orig_col = "Label"


# =========================
# Utilities
# =========================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def stable_hash_text(s: str) -> str:
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return h

def shuffle_df(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

def normalize_text_basic(text: str) -> str:
    if pd.isna(text):
        return ""
    s = str(text)
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("Â", "")
    s = s.replace("’", "'").replace("‘", "'").replace("`", "'")
    s = s.replace("“", '"').replace("”", '"')
    s = re.sub(r"\s+", " ", s).strip()
    return s

def map_labels_explicit(label_series: pd.Series) -> Dict[Any, int]:
    vals = set(label_series.dropna().astype(str).str.strip().str.upper().unique().tolist())
    if vals != {"UGE", "SE"}:
        raise ValueError(f"Unexpected labels in data: {vals}. Expected {{'SE','UGE'}}")
    return {"SE": 0, "UGE": 1}


# =========================
# UGE extraction for disjointness
# =========================

def strip_uge_tags_keep_content(text: str) -> str:
    if pd.isna(text):
        return ""
    return UGE_TAG_PATTERN.sub(lambda m: m.group(1), str(text))

def extract_uge_items_as_list(text: str) -> List[str]:
    if pd.isna(text):
        return []
    items = UGE_TAG_PATTERN.findall(str(text))
    cleaned = []
    for it in items:
        it2 = normalize_text_basic(it)
        if it2:
            cleaned.append(it2)
    return cleaned

def normalize_span_canonical(span: str) -> str:
    s = normalize_text_basic(span).lower()
    s = re.sub(r"[^a-z0-9\s']+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def simple_stem(token: str) -> str:
    t = token.lower()
    t = re.sub(r"[^a-z0-9']+", "", t)
    t = t.strip("'")
    if len(t) <= 3:
        return t
    if t.endswith("ies") and len(t) > 4:
        t = t[:-3] + "y"
    elif t.endswith("s") and not t.endswith("ss"):
        t = t[:-1]
    for suf in ["ing", "ed", "ly", "er", "ers", "est", "ness", "ment", "tion", "tions"]:
        if t.endswith(suf) and len(t) > len(suf) + 2:
            t = t[: -len(suf)]
            break
    return t

def tokenize_normalized(span: str) -> Set[str]:
    s = normalize_span_canonical(span)
    toks = re.findall(r"[a-z0-9']+", s)
    stems = set()
    for tok in toks:
        if tok:
            stems.add(simple_stem(tok))
    return stems

def extract_disjoint_units_from_text(text: str, unit: str) -> Set[str]:
    spans = extract_uge_items_as_list(text)
    if unit == "span":
        return {normalize_span_canonical(sp) for sp in spans if normalize_span_canonical(sp)}
    if unit == "token":
        out = set()
        for sp in spans:
            out |= tokenize_normalized(sp)
        return out
    raise ValueError(f"Unknown DISJOINT_UNIT={unit}. Use 'span' or 'token'.")

def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# =========================
# Experiments: model inputs
# =========================
UGE_START = "<UGE_SPAN>"
UGE_END   = "</UGE_SPAN>"
CUE_PLACEHOLDER = "[CUE]"

def build_sentence_processed_for_experiment_1(raw_sentence: str) -> str:
    raw_sentence = normalize_text_basic(raw_sentence)

    def repl(m):
        content = normalize_text_basic(m.group(1))
        return f" {UGE_START} {content} {UGE_END} "

    localized = UGE_TAG_PATTERN.sub(repl, raw_sentence)
    localized = re.sub(r"\s+", " ", localized).strip()
    return localized

def build_sentence_processed_for_experiment_2(raw_sentence: str) -> str:
    raw_sentence = normalize_text_basic(raw_sentence)
    sentence_wo_tags = strip_uge_tags_keep_content(raw_sentence)
    extracted_items = extract_uge_items_as_list(raw_sentence)
    sentence_wo_tags = normalize_text_basic(sentence_wo_tags)

    if not extracted_items:
        return sentence_wo_tags.strip()

    aux_text = "UGE_EXTRACTED: " + " ; ".join(extracted_items)
    return f"{sentence_wo_tags} [SEP] {aux_text}".strip()

def replace_uge_spans_with_placeholder(text: str, placeholder: str = CUE_PLACEHOLDER) -> str:
    if pd.isna(text):
        return ""
    t = normalize_text_basic(text)

    def repl(_m):
        return f" {placeholder} "

    t2 = UGE_TAG_PATTERN.sub(repl, t)
    t2 = re.sub(r"\s+", " ", t2).strip()
    return t2

def build_sentence_processed_for_experiment_3(raw_sentence: str) -> str:
    return replace_uge_spans_with_placeholder(raw_sentence, placeholder=CUE_PLACEHOLDER)

def prepare_dataframe_for_experiment(df_in: pd.DataFrame, experiment: int) -> pd.DataFrame:
    df = df_in.copy()
    df["sentence"] = df["sentence"].apply(normalize_text_basic)

    if experiment == 1:
        df["sentence_processed"] = df["sentence"].apply(build_sentence_processed_for_experiment_1)
    elif experiment == 2:
        df["sentence_processed"] = df["sentence"].apply(build_sentence_processed_for_experiment_2)
    elif experiment == 3:
        df["sentence_processed"] = df["sentence"].apply(build_sentence_processed_for_experiment_3)
    else:
        raise ValueError("experiment must be 1, 2, or 3")

    return df


# =========================
# Disjointness enforcement + leakage checks
# =========================

def drop_exact_duplicates_across_splits(df_trainval: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_hashes = set(df_trainval["sentence"].map(stable_hash_text).tolist())
    keep_mask = ~df_test["sentence"].map(stable_hash_text).isin(train_hashes)
    df_test_filtered = df_test.loc[keep_mask].reset_index(drop=True)
    df_test_dropped = df_test.loc[~keep_mask].reset_index(drop=True)
    return df_test_filtered, df_test_dropped

def enforce_disjoint_uge_test_cue(df_trainval: pd.DataFrame, df_test: pd.DataFrame, unit: str, action: str):
    df_trainval_norm = df_trainval.copy()
    df_test_norm = df_test.copy()
    df_trainval_norm["sentence"] = df_trainval_norm["sentence"].apply(normalize_text_basic)
    df_test_norm["sentence"] = df_test_norm["sentence"].apply(normalize_text_basic)

    seen_cues_set: Set[str] = set()
    seen_cues_per_row: List[Set[str]] = []

    for txt in df_trainval_norm["sentence"].astype(str).tolist():
        cue_units = extract_disjoint_units_from_text(txt, unit=unit)
        seen_cues_set |= cue_units
        seen_cues_per_row.append(cue_units)

    dropped_rows = []
    keep_mask = []

    for i, txt in enumerate(df_test_norm["sentence"].astype(str).tolist()):
        test_cues = extract_disjoint_units_from_text(txt, unit=unit)

        if not DISJOINT_APPROXIMATE:
            overlaps = len(test_cues.intersection(seen_cues_set)) > 0
        else:
            best_sim = 0.0
            for ref in seen_cues_per_row:
                best_sim = max(best_sim, jaccard(test_cues, ref))
                if best_sim >= DISJOINT_JACCARD_THRESHOLD:
                    break
            overlaps = best_sim >= DISJOINT_JACCARD_THRESHOLD

        if overlaps and action == "drop":
            dropped_rows.append(i)
            keep_mask.append(False)
        else:
            keep_mask.append(True)

    df_test_filtered = df_test_norm.iloc[keep_mask].reset_index(drop=True)
    df_test_dropped = df_test_norm.iloc[dropped_rows].reset_index(drop=True) if dropped_rows else df_test_norm.iloc[0:0].copy()
    return df_test_filtered, df_test_dropped

def enforce_disjoint_uge_test_embedding(df_trainval: pd.DataFrame, df_test: pd.DataFrame, action: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise RuntimeError("DISJOINT_MODE='embedding' requires: pip install sentence-transformers") from e

    df_trainval_norm = df_trainval.copy()
    df_test_norm = df_test.copy()
    df_trainval_norm["sentence"] = df_trainval_norm["sentence"].apply(normalize_text_basic)
    df_test_norm["sentence"] = df_test_norm["sentence"].apply(normalize_text_basic)

    model = SentenceTransformer(DISJOINT_EMBEDDING_MODEL)

    trainval_texts = df_trainval_norm["sentence"].astype(str).tolist()
    test_texts = df_test_norm["sentence"].astype(str).tolist()

    trainval_emb = model.encode(
        trainval_texts,
        batch_size=DISJOINT_EMBEDDING_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    test_emb = model.encode(
        test_texts,
        batch_size=DISJOINT_EMBEDDING_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    dropped_rows = []
    keep_mask = []

    for i in range(len(test_texts)):
        best_sim = float(np.max(np.dot(trainval_emb, test_emb[i])))
        overlaps = best_sim >= DISJOINT_EMBEDDING_THRESHOLD

        if overlaps and action == "drop":
            dropped_rows.append(i)
            keep_mask.append(False)
        else:
            keep_mask.append(True)

    df_test_filtered = df_test_norm.iloc[keep_mask].reset_index(drop=True)
    df_test_dropped = df_test_norm.iloc[dropped_rows].reset_index(drop=True) if dropped_rows else df_test_norm.iloc[0:0].copy()
    return df_test_filtered, df_test_dropped

def compute_trainval_test_cue_overlap(df_trainval: pd.DataFrame, df_test: pd.DataFrame, unit: str) -> Dict[str, Any]:
    train_cues = set()
    for txt in df_trainval["sentence"].astype(str).tolist():
        train_cues |= extract_disjoint_units_from_text(txt, unit=unit)

    test_rows_overlap = []
    test_any_overlap = 0
    for i, txt in enumerate(df_test["sentence"].astype(str).tolist()):
        test_cues = extract_disjoint_units_from_text(txt, unit=unit)
        overlaps = test_cues.intersection(train_cues)
        if overlaps:
            test_any_overlap += 1
            test_rows_overlap.append({"test_idx": i, "overlap_size": len(overlaps)})

    return {
        "train_cue_vocab_size": len(train_cues),
        "test_rows_with_overlap": test_any_overlap,
        "test_total_rows": len(df_test),
        "sample_overlapping_rows": test_rows_overlap[:20],
    }


# =========================
# Dataset wrapper
# =========================
class TextDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx]).long()
        return item


# =========================
# Plots / metrics
# =========================
def plot_confusion_matrix(cm: np.ndarray, out_path: str, title: str):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["StE (0)", "UgE (1)"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_roc_curves(all_model_roc_data: Dict[str, Dict[str, Any]], out_path: str, title: str):
    plt.figure(figsize=(8, 6))
    for model_name, roc_data in all_model_roc_data.items():
        plt.plot(
            roc_data["fpr"],
            roc_data["tpr"],
            lw=2,
            label=f"{model_name} (AUC={roc_data['auc']:.3f})",
        )
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_training_curves_from_log_history(log_history: List[Dict[str, Any]], out_path_prefix: str, model_name: str):
    train_loss, eval_loss, train_acc, eval_acc = [], [], [], []
    for entry in log_history:
        step = entry.get("step", None)
        if step is None:
            continue
        if "loss" in entry:
            train_loss.append((step, entry["loss"]))
        if "eval_loss" in entry:
            eval_loss.append((step, entry["eval_loss"]))
        if "accuracy" in entry:
            train_acc.append((step, entry["accuracy"]))
        if "eval_accuracy" in entry:
            eval_acc.append((step, entry["eval_accuracy"]))

    plt.figure(figsize=(8, 6))
    if train_loss:
        xs, ys = zip(*train_loss)
        plt.plot(xs, ys, label="Train Loss")
    if eval_loss:
        xs, ys = zip(*eval_loss)
        plt.plot(xs, ys, label="Validation Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title(f"{model_name} - Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_path_prefix}_loss.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 6))
    if train_acc:
        xs, ys = zip(*train_acc)
        plt.plot(xs, ys, label="Train Accuracy")
    if eval_acc:
        xs, ys = zip(*eval_acc)
        plt.plot(xs, ys, label="Validation Accuracy")
    plt.xlabel("Steps")
    plt.ylabel("Accuracy")
    plt.title(f"{model_name} - Training vs Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_path_prefix}_accuracy.png", dpi=200)
    plt.close()


def _get_eval_strategy_key():
    sig = inspect.signature(TrainingArguments.__init__).parameters
    if "evaluation_strategy" in sig:
        return "evaluation_strategy"
    if "eval_strategy" in sig:
        return "eval_strategy"
    raise ValueError("Could not find evaluation strategy key in TrainingArguments.")


# =========================
# Robust input sanitization / checks (ADDED)
# =========================

def _sanitize_text_value(x: Any) -> str:
    # Convert NaN/None to empty string; everything else to string.
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    # Also handle pandas NA
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x)

def sanitize_text_list(xs: List[Any]) -> List[str]:
    return [_sanitize_text_value(x) for x in xs]

def debug_split_inputs(prefix: str, X_texts: List[str], y: List[int], max_debug: int = 3):
    # y can be list of ints or numpy array
    xlen = 0 if X_texts is None else len(X_texts)
    ylen = 0 if y is None else len(y)

    print(f"[DEBUG INPUTS] {prefix}: len(X)={xlen}, len(y)={ylen}")
    if X_texts is not None and len(X_texts) > 0:
        empties = sum(1 for t in X_texts if t == "")
        non_str = sum(1 for t in X_texts if not isinstance(t, str))
        print(f"[DEBUG INPUTS] {prefix}: empty_texts={empties}, non_str_entries={non_str}")
        print(f"[DEBUG INPUTS] {prefix}: sample_texts={X_texts[:max_debug]}")
    if y is not None and len(y) > 0:
        uniq = sorted(set(int(v) for v in y))
        print(f"[DEBUG INPUTS] {prefix}: y_unique={uniq}")


def assert_valid_split(prefix: str, X_texts: List[str], y: List[int]):
    if X_texts is None or len(X_texts) == 0:
        raise RuntimeError(f"[ASSERT FAIL] {prefix}: X_texts is empty.")
    if y is None or len(y) == 0:
        raise RuntimeError(f"[ASSERT FAIL] {prefix}: y is empty.")
    if len(X_texts) != len(y):
        raise RuntimeError(f"[ASSERT FAIL] {prefix}: len(X_texts) != len(y). got {len(X_texts)} vs {len(y)}")


# =========================
# Training runner
# =========================
def run_for_single_model(
    model_display_name: str,
    model_name_or_path: str,
    X_train_texts: List[str],
    y_train: List[int],
    X_val_texts: List[str],
    y_val: List[int],
    X_test_texts: List[str],
    y_test: List[int],
    label_to_int: Dict[Any, int],
    experiment_dir: str,
    experiment_tag: str,
    model_experiment_id: str,
    sentence_original_test: List[Any],
    label_original_test: List[Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:

    # --------- ADDED checks/sanitization ----------
    X_train_texts = sanitize_text_list(X_train_texts)
    X_val_texts = sanitize_text_list(X_val_texts)
    X_test_texts = sanitize_text_list(X_test_texts)
    y_train = [int(v) for v in y_train]
    y_val = [int(v) for v in y_val]
    y_test = [int(v) for v in y_test]

    debug_split_inputs("train", X_train_texts, y_train)
    debug_split_inputs("val", X_val_texts, y_val)
    debug_split_inputs("test", X_test_texts, y_test)

    assert_valid_split("train", X_train_texts, y_train)
    assert_valid_split("val", X_val_texts, y_val)
    assert_valid_split("test", X_test_texts, y_test)

    # Sanity: tokenizer should get at least one non-empty string; if all are empty,
    # fast tokenizers can behave poorly.
    if sum(1 for t in X_train_texts if t.strip() != "") == 0:
        raise RuntimeError(f"[ASSERT FAIL] {model_display_name}: all train texts are empty strings.")
    if sum(1 for t in X_val_texts if t.strip() != "") == 0:
        raise RuntimeError(f"[ASSERT FAIL] {model_display_name}: all val texts are empty strings.")
    if sum(1 for t in X_test_texts if t.strip() != "") == 0:
        raise RuntimeError(f"[ASSERT FAIL] {model_display_name}: all test texts are empty strings.")
    # ------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    config = AutoConfig.from_pretrained(model_name_or_path)
    config.num_labels = 2
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=config)

    # Extra try/except wrapper around tokenization to print offending lengths/content
    try:
        train_enc = tokenizer(X_train_texts, truncation=True, padding=True, max_length=MAX_LENGTH)
        val_enc = tokenizer(X_val_texts, truncation=True, padding=True, max_length=MAX_LENGTH)
        test_enc = tokenizer(X_test_texts, truncation=True, padding=True, max_length=MAX_LENGTH)
    except Exception as e:
        print("\n[ERROR] Tokenization failed.")
        print("Model:", model_display_name, model_name_or_path)
        print("Lengths:",
              "train", len(X_train_texts), "val", len(X_val_texts), "test", len(X_test_texts))
        print("Sample train texts:", X_train_texts[:5])
        print("Sample val texts:", X_val_texts[:5])
        print("Sample test texts:", X_test_texts[:5])
        raise

    train_dataset = TextDataset(train_enc, y_train)
    val_dataset = TextDataset(val_enc, y_val)
    test_dataset = TextDataset(test_enc, y_test)

    def compute_metrics(eval_pred):
        preds = eval_pred.predictions if hasattr(eval_pred, "predictions") else eval_pred[0]
        labels = eval_pred.label_ids if hasattr(eval_pred, "label_ids") else eval_pred[1]

        if isinstance(preds, (tuple, list)):
            preds = preds[0]
        preds = np.asarray(preds)

        if preds.ndim != 2 or preds.shape[1] != 2:
            raise ValueError(f"Unexpected prediction shape: {preds.shape}; expected (N, 2).")

        y_pred = np.argmax(preds, axis=-1)
        acc = accuracy_score(labels, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(labels, y_pred, average="binary", zero_division=0)
        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

    run_name = f"{experiment_tag}_{model_display_name}".replace(" ", "_")
    eval_strategy_key = _get_eval_strategy_key()

    training_kwargs = {
        "output_dir": os.path.join(experiment_dir, "hf_runs", run_name),
        "learning_rate": LEARNING_RATE,
        "lr_scheduler_type": "linear",
        "per_device_train_batch_size": BATCH_SIZE,
        "per_device_eval_batch_size": BATCH_SIZE,
        "num_train_epochs": EPOCHS,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "logging_steps": LOGGING_STEPS,
        "logging_strategy": "steps",
        "report_to": [],
        "seed": SEED,
        "metric_for_best_model": "eval_f1",
        "greater_is_better": True,
        "load_best_model_at_end": False,
        "save_strategy": SAVE_STRATEGY,
    }

    training_kwargs[eval_strategy_key] = "epoch"
    training_args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=default_data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print(f"\nTraining: {model_display_name} | {experiment_tag} | id={model_experiment_id}")
    trainer.train()

    # ---- get probabilities on val for threshold tuning ----
    val_out = trainer.predict(val_dataset)
    val_logits = val_out.predictions[0] if isinstance(val_out.predictions, (tuple, list)) else val_out.predictions
    val_logits = np.asarray(val_logits)
    val_probs = torch.softmax(torch.tensor(val_logits), dim=-1).numpy()
    val_score = val_probs[:, 1]
    val_y_true = np.array(y_val, dtype=int)

    # ---- get probabilities on test ----
    test_out = trainer.predict(test_dataset)
    test_logits = test_out.predictions[0] if isinstance(test_out.predictions, (tuple, list)) else test_out.predictions
    test_logits = np.asarray(test_logits)
    test_probs = torch.softmax(torch.tensor(test_logits), dim=-1).numpy()
    test_score = test_probs[:, 1]

    int_to_label = {v: k for k, v in label_to_int.items()}

    # ROC (independent of threshold)
    fpr, tpr, _ = roc_curve(np.array(y_test, dtype=int), test_score)
    roc_auc = auc(fpr, tpr)
    roc_data = {"fpr": fpr, "tpr": tpr, "auc": roc_auc}

    # evaluate at two regimes: tuned threshold & fixed 0.5
    thresholds_to_report = []
    if THRESHOLD_TUNING_ON_VAL:
        thresholds = np.linspace(0.05, 0.95, 19)
        best_t, best_f1 = 0.5, -1.0
        for t in thresholds:
            val_y_pred = (val_score >= t).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(val_y_true, val_y_pred, average="binary", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        thresholds_to_report.append(("val_tuned", best_t))
    if REPORT_FIXED_THRESHOLD_0_5:
        thresholds_to_report.append(("fixed_0.5", 0.5))

    results_rows = []
    y_true = np.array(y_test, dtype=int)
    y_true_label_original = np.array([int_to_label.get(int(ti), "") for ti in y_true], dtype=object)

    for tag, t in thresholds_to_report:
        y_pred = (test_score >= t).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        acc = accuracy_score(y_true, y_pred)

        y_pred_label_original = np.array([int_to_label.get(int(pi), "") for pi in y_pred], dtype=object)

        results_rows.append({
            "Model": model_display_name,
            "ThresholdTag": tag,
            "ThresholdValue": float(t),
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "Accuracy": acc,
            "ROC_AUC": roc_auc,
        })

        df_pred_out = pd.DataFrame({
            "test_row_id": np.arange(len(y_true), dtype=int),
            "sentence_original": np.array(sentence_original_test, dtype=object).astype(str),
            "label_original": np.array(label_original_test, dtype=object).astype(str),
            "y_true_mapped": y_true.astype(int),
            "y_pred_mapped": y_pred.astype(int),
            "y_pred_label_original": y_pred_label_original,
            "y_true_label_original": y_true_label_original,
            "prob_class_0": test_probs[:, 0],
            "prob_class_1": test_probs[:, 1],
            "ThresholdTag": tag,
            "ThresholdValue": float(t),
            "experiment": os.path.basename(os.path.dirname(experiment_dir)),
            "experiment_tag": experiment_tag,
            "model": model_display_name,
            "shard_id": SHARD_ID,
        })

        pred_csv_path = os.path.join(
            experiment_dir,
            f"predictions_{experiment_tag}_{model_display_name}__{tag}.csv"
        )
        df_pred_out.to_csv(pred_csv_path, index=False)

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(int)
        cm_path = os.path.join(experiment_dir, f"confusion_matrix_{model_display_name}__{tag}.png")
        plot_confusion_matrix(cm, cm_path, f"{experiment_tag} | {model_display_name} | {tag}")

    log_history = trainer.state.log_history
    curves_prefix = os.path.join(experiment_dir, f"training_curves_{model_display_name}")
    plot_training_curves_from_log_history(log_history, curves_prefix, model_display_name)

    metrics_csv_path = os.path.join(experiment_dir, f"metrics_{experiment_tag}_{model_display_name}.csv")
    pd.DataFrame(results_rows).to_csv(metrics_csv_path, index=False)

    return results_rows, roc_data


# =========================
# Sharding
# =========================
def shard_dataframe(df: pd.DataFrame, num_shards: int, shard_id: int) -> pd.DataFrame:
    if num_shards <= 1:
        return df
    df = df.reset_index(drop=True)
    mask = (np.arange(len(df)) % num_shards) == shard_id
    return df.loc[mask].reset_index(drop=True)


# =========================
# Main experiment
# =========================
def run_experiment(experiment: int):
    df_train_full = pd.read_csv(TRAIN_DATA_PATH, encoding="Latin1")
    df_test_full  = pd.read_csv(TEST_DATA_PATH, encoding="Latin1")

    df_train_full[lab_orig_col] = df_train_full[lab_orig_col].astype(str).str.strip().str.upper()
    df_test_full[lab_orig_col]  = df_test_full[lab_orig_col].astype(str).str.strip().str.upper()

    label_to_int = map_labels_explicit(df_train_full[lab_orig_col])

    df_train_full["Label_mapped"] = df_train_full[lab_orig_col].map(label_to_int).astype(int)
    df_test_full["Label_mapped"] = df_test_full[lab_orig_col].map(label_to_int).astype(int)

    from sklearn.model_selection import train_test_split
    df_train_full = shuffle_df(df_train_full, seed=SEED)

    idx = np.arange(len(df_train_full))
    train_idx, val_idx = train_test_split(
        idx,
        test_size=0.10,
        random_state=SEED,
        stratify=df_train_full["Label_mapped"].astype(int).values,
        shuffle=True,
    )

    df_train = df_train_full.iloc[train_idx].reset_index(drop=True)
    df_val   = df_train_full.iloc[val_idx].reset_index(drop=True)
    df_trainval = pd.concat([df_train, df_val], axis=0).reset_index(drop=True)

    df_trainval["sentence"] = df_trainval["sentence"].apply(normalize_text_basic)
    df_test_full["sentence"] = df_test_full["sentence"].apply(normalize_text_basic)

    if DROP_EXACT_DUPLICATES_ACROSS_SPLITS:
        df_test_filtered, df_test_dropped_dup = drop_exact_duplicates_across_splits(df_trainval, df_test_full)
    else:
        df_test_filtered = df_test_full.copy().reset_index(drop=True)
        df_test_dropped_dup = df_test_full.iloc[0:0].copy()

    if DISJOINT_MODE == "cue":
        df_test_filtered2, df_test_dropped = enforce_disjoint_uge_test_cue(
            df_trainval=df_trainval,
            df_test=df_test_filtered,
            unit=DISJOINT_UNIT,
            action=DISJOINT_TEST_ACTION,
        )
        df_test_dropped = pd.concat([df_test_dropped_dup, df_test_dropped], axis=0).reset_index(drop=True)
        df_test_filtered = df_test_filtered2
    elif DISJOINT_MODE == "embedding":
        df_test_filtered2, df_test_dropped = enforce_disjoint_uge_test_embedding(
            df_trainval=df_trainval,
            df_test=df_test_filtered,
            action=DISJOINT_TEST_ACTION
        )
        df_test_dropped = pd.concat([df_test_dropped_dup, df_test_dropped], axis=0).reset_index(drop=True)
        df_test_filtered = df_test_filtered2
    elif DISJOINT_MODE == "none":
        df_test_dropped = df_test_filtered.iloc[0:0].copy()
    else:
        raise ValueError("DISJOINT_MODE must be cue | embedding | none")

    if DISJOINT_UNIT in ("span", "token"):
        overlap_stats = compute_trainval_test_cue_overlap(df_trainval, df_test_filtered, unit=DISJOINT_UNIT)
        overlap_msg = (
            f"[LeakageCheck] DISJOINT_MODE={DISJOINT_MODE} unit={DISJOINT_UNIT} "
            f"trainCueVocab={overlap_stats['train_cue_vocab_size']} "
            f"testRowsWithOverlap={overlap_stats['test_rows_with_overlap']}/{overlap_stats['test_total_rows']}"
        )
        print(overlap_msg)

        if FAIL_IF_TRAINVAL_TEST_CUE_OVERLAP and overlap_stats["test_rows_with_overlap"] > 0:
            raise RuntimeError(
                "Leakage verification failed: train/val cue overlap still exists in the filtered test. "
                f"Details: {overlap_stats['sample_overlapping_rows'][:5]}"
            )

    df_train = shuffle_df(df_train, seed=SEED + 2)
    df_val = shuffle_df(df_val, seed=SEED + 3)
    df_test_filtered = shuffle_df(df_test_filtered, seed=SEED + 4)

    print(f"[SplitSizes] train={len(df_train)} val={len(df_val)} test_original={len(df_test_full)} test_kept={len(df_test_filtered)}")

    experiment_tag = f"Experiment_{experiment}_shard{SHARD_ID}of{NUM_SHARDS}__leakageSafe__disjointMode_{DISJOINT_MODE}"
    if DISJOINT_MODE == "cue":
        experiment_tag += f"__unit_{DISJOINT_UNIT}__approx{int(DISJOINT_APPROXIMATE)}__jthr{DISJOINT_JACCARD_THRESHOLD}"
    elif DISJOINT_MODE == "embedding":
        experiment_tag += f"__embedModel_{DISJOINT_EMBEDDING_MODEL}__thr{DISJOINT_EMBEDDING_THRESHOLD}"
    experiment_tag += f"__test_{DISJOINT_TEST_ACTION}"

    exp_dir = os.path.join(OUTPUT_BASE_DIR, f"Experiment_{experiment}_sharded", experiment_tag)
    ensure_dir(exp_dir)
    ensure_dir(os.path.join(exp_dir, "hf_runs"))

    df_train_processed = prepare_dataframe_for_experiment(df_train, experiment=experiment)
    df_val_processed   = prepare_dataframe_for_experiment(df_val, experiment=experiment)
    df_test_processed  = prepare_dataframe_for_experiment(df_test_filtered, experiment=experiment)

    df_train_part = shard_dataframe(df_train_processed, num_shards=NUM_SHARDS, shard_id=SHARD_ID)

    # --------- ADDED: skip empty shard early ----------
    if len(df_train_part) == 0:
        print(f"[SKIP] experiment={experiment} shard_id={SHARD_ID}/{NUM_SHARDS}: empty train shard after sharding.")
        return
    # ---------------------------------------------------

    # Prepare arrays
    X_train_texts = df_train_part["sentence_processed"].astype(str).tolist()
    y_train = df_train_part["Label_mapped"].astype(int).tolist()

    X_val_texts = df_val_processed["sentence_processed"].astype(str).tolist()
    y_val = df_val_processed["Label_mapped"].astype(int).tolist()

    X_test_texts = df_test_processed["sentence_processed"].astype(str).tolist()
    y_test = df_test_processed["Label_mapped"].astype(int).tolist()

    sentence_original_test = df_test_processed[sent_orig_col].astype(str).tolist()
    label_original_test = df_test_processed[lab_orig_col].astype(str).tolist()

    diag = [
        {"split": "train", "rows": len(df_train)},
        {"split": "val", "rows": len(df_val)},
        {"split": "test_kept", "rows": len(df_test_filtered)},
        {"split": "train_shard_kept", "rows": len(df_train_part)},
        {"split": "FAIL_IF_TRAINVAL_TEST_CUE_OVERLAP", "rows": int(FAIL_IF_TRAINVAL_TEST_CUE_OVERLAP)},
        {"split": "DROP_EXACT_DUPLICATES_ACROSS_SPLITS", "rows": int(DROP_EXACT_DUPLICATES_ACROSS_SPLITS)},
    ]
    pd.DataFrame(diag).to_csv(os.path.join(exp_dir, "disjointness_diagnostics.csv"), index=False)

    # Extra: if val/test become empty, fail fast
    if len(X_val_texts) == 0 or len(X_test_texts) == 0:
        raise RuntimeError(
            f"Empty split after processing: val={len(X_val_texts)}, test={len(X_test_texts)}. "
            f"Check disjoint filtering settings."
        )

    all_metrics = []
    all_roc_data = {}

    for model_display_name, model_name_or_path in MODEL_NAMES.items():
        model_id = f"{experiment_tag}__{model_display_name}"
        print(f"\n=== Training model: {model_display_name} | {experiment_tag} ===")

        results_rows, roc_data = run_for_single_model(
            model_display_name=model_display_name,
            model_name_or_path=model_name_or_path,
            X_train_texts=X_train_texts,
            y_train=y_train,
            X_val_texts=X_val_texts,
            y_val=y_val,
            X_test_texts=X_test_texts,
            y_test=y_test,
            label_to_int=label_to_int,
            experiment_dir=exp_dir,
            experiment_tag=experiment_tag,
            model_experiment_id=model_id,
            sentence_original_test=sentence_original_test,
            label_original_test=label_original_test,
        )
        all_metrics.extend(results_rows)
        all_roc_data[model_display_name] = roc_data

    metrics_all_path = os.path.join(exp_dir, f"all_models_metrics_{experiment_tag}.csv")
    pd.DataFrame(all_metrics).to_csv(metrics_all_path, index=False)

    roc_plot_path = os.path.join(exp_dir, f"ROC_curves_all_models_{experiment_tag}.png")
    plot_roc_curves(all_roc_data, roc_plot_path, f"ROC Curves - All Models ({experiment_tag})")

    print(f"\nDone {experiment_tag}. Outputs saved to: {exp_dir}")


# =========================
# Main
# =========================
if __name__ == "__main__":
    ensure_dir(OUTPUT_BASE_DIR)

    print("Transformers version:", transformers.__version__)
    print(f"Sharding: NUM_SHARDS={NUM_SHARDS}, SHARD_ID={SHARD_ID}")
    print(f"MERGE_ONLY={MERGE_ONLY}")

    if MERGE_ONLY:
        raise RuntimeError("MERGE_ONLY is not supported in this leakage-safe version.")

    print("Disjointness/leakage settings:",
          f"DISJOINT_MODE={DISJOINT_MODE}, DISJOINT_UNIT={DISJOINT_UNIT}, DISJOINT_TEST_ACTION={DISJOINT_TEST_ACTION}, "
          f"FAIL_IF_TRAINVAL_TEST_CUE_OVERLAP={FAIL_IF_TRAINVAL_TEST_CUE_OVERLAP}, "
          f"DROP_EXACT_DUPLICATES_ACROSS_SPLITS={DROP_EXACT_DUPLICATES_ACROSS_SPLITS}, "
          f"THRESHOLD_TUNING_ON_VAL={THRESHOLD_TUNING_ON_VAL}, REPORT_FIXED_THRESHOLD_0_5={REPORT_FIXED_THRESHOLD_0_5}")

    for exp in (1, 2, 3):
        run_experiment(experiment=exp)

    print("\nALL DONE.")

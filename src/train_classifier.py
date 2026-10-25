from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import label_binarize

# Sentence-Transformers
from sentence_transformers import SentenceTransformer

from utils import (
    read_data,
    read_data_skill_specific,
    eval_at_threshold,
    pick_threshold_precision,
    summarize_latency,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SUPPORTED_MODELS = ["logistic_regression", "svm", "naive_bayes", "decision_tree", "random_forest"]
COLLAPSED_SKILL_LABELS = ["SKILL", "OOD", "UNCLEAR"]


def normalize_model_id(model_name: str) -> str:
    # Safe filename key
    return model_name.replace("/", "__").replace(":", "_")


def normalize_variant_id(variant: str | None, fallback: str = "default") -> str:
    value = (variant or fallback).strip().lower()
    if not value:
        value = fallback
    return value.replace(" ", "_").replace("/", "_")


def save_run_artifacts(
    *,
    output_dir: str,
    model_name: str,
    task: str,
    variant: str,
    predictions_df: pd.DataFrame,
    results: Dict[str, object],
) -> tuple[Path, Path]:
    run_dir = Path(output_dir) / normalize_model_id(model_name) / task / variant
    run_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = run_dir / "predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    logger.info("Saved predictions to %s", predictions_path)

    results_path = run_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Saved results to %s", results_path)

    return predictions_path, results_path


def embed_texts(
    model_name: str,
    texts: List[str],
    batch_size: int = 64,
    cache_dir: str | None = "../models/embeddings_cache",
    task: str = None,
) -> np.ndarray:
    """
    Embed texts using SentenceTransformer with optional disk caching.
    """
    # jina-embeddings-v3 ships with an optional flash-attention backend
    # (xlm-roberta-flash-implementation) that requires compiled CUDA kernels.
    # Force the standard eager implementation to avoid missing-file errors.
    # Also patch transformers 5.x compatibility: XLMRobertaLoRA (Jina's custom class)
    # does not define `all_tied_weights_keys`, which transformers>=5.0 now requires.
    model_kwargs = {}
    if "jina" in model_name.lower():
        model_kwargs["attn_implementation"] = "eager"
        import transformers as _transformers
        _orig_model_init = _transformers.PreTrainedModel.__init__
        def _patched_model_init(self, *args, **kwargs):
            _orig_model_init(self, *args, **kwargs)
            if not hasattr(self, "all_tied_weights_keys"):
                self.all_tied_weights_keys = []
        _transformers.PreTrainedModel.__init__ = _patched_model_init

    st_model = SentenceTransformer(model_name, cache_folder=cache_dir, trust_remote_code=True, model_kwargs=model_kwargs)

    # Encode
    embeddings = st_model.encode(
        texts,
        task=task,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # cosine-friendly & tends to help linear classifiers
    )

    return embeddings


def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    model_type: str = "logistic_regression",
    C: float = 1.0,
    class_weight: str | None = "balanced",
    calibrate: str = "none",
    random_state: int = 42,
) -> object:
    """
    Train a classifier; optionally calibrate probabilities.

    model_type: "logistic_regression" | "svm" | "random_forest"
    calibrate:  "none" | "sigmoid" | "isotonic"

    Notes
    -----
    - LinearSVC does not produce probabilities natively and is always wrapped with
      CalibratedClassifierCV regardless of the calibrate setting (fallback: "sigmoid").
    """
    if model_type not in SUPPORTED_MODELS:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose from {SUPPORTED_MODELS}")

    def calibration_cv() -> int:
        _, class_counts = np.unique(y_train, return_counts=True)
        min_class_count = int(class_counts.min())
        if min_class_count < 2:
            raise ValueError(
                "Probability calibration needs at least two training samples per class. "
                "Lower test_size, drop rare classes with min_samples_per_class, or use a non-SVM model "
                "with calibrate='none'."
            )
        cv = min(3, min_class_count)
        if cv < 3:
            logger.info(
                "Using %d-fold calibration because the smallest training class has %d samples.",
                cv,
                min_class_count,
            )
        return cv

    if model_type == "logistic_regression":
        base = LogisticRegression(
            C=C,
            max_iter=2000,
            solver="lbfgs",
            class_weight=class_weight,
            random_state=random_state,
        )
    elif model_type == "svm":
        # LinearSVC is faster for high-dimensional embeddings; calibrate for probabilities
        base = LinearSVC(
            C=C,
            max_iter=2000,
            class_weight=class_weight,
            random_state=random_state,
        )
        cal_method = calibrate if calibrate != "none" else "sigmoid"
        cal = CalibratedClassifierCV(base, method=cal_method, cv=calibration_cv())
        cal.fit(X_train, y_train)
        return cal
    elif model_type == "random_forest":
        base = RandomForestClassifier(
            n_estimators=200,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_type == "naive_bayes":
        base = GaussianNB()
    elif model_type == "decision_tree":
        base = DecisionTreeClassifier(
            class_weight=class_weight,
            random_state=random_state,
        )

    if calibrate == "none":
        base.fit(X_train, y_train)
        return base

    # Calibration needs held-out folds; use up to 3 folds based on the smallest class.
    cal = CalibratedClassifierCV(base, method=calibrate, cv=calibration_cv())
    cal.fit(X_train, y_train)
    return cal


def predict_proba_positive(clf: object, X: np.ndarray) -> np.ndarray:
    proba = clf.predict_proba(X)
    return proba[:, 1]


def predict_proba_with_latency(clf: object, X: np.ndarray) -> tuple[np.ndarray, Dict[str, object], np.ndarray]:
    """Predict one embedded sample at a time and summarize classifier latency."""
    proba_rows = []
    latencies_seconds = []

    for row in X:
        sample = row.reshape(1, -1)
        start = time.perf_counter()
        proba = clf.predict_proba(sample)
        latencies_seconds.append(time.perf_counter() - start)
        proba_rows.append(proba[0])

    if proba_rows:
        proba_matrix = np.vstack(proba_rows)
    else:
        proba_matrix = np.empty((0, 0), dtype=np.float64)

    return (
        proba_matrix,
        summarize_latency(latencies_seconds, measurement="per_sample_predict_proba"),
        np.asarray(latencies_seconds, dtype=np.float64),
    )


def predict_proba_positive_with_latency(clf: object, X: np.ndarray) -> tuple[np.ndarray, Dict[str, object], np.ndarray]:
    proba, latency_stats, latencies_seconds = predict_proba_with_latency(clf, X)
    return proba[:, 1], latency_stats, latencies_seconds


def _collapse_skill_specific_label(label: str) -> str:
    if label == "out_of_scope":
        return "OOD"
    if label == "unclear":
        return "UNCLEAR"
    return "SKILL"


def _drop_rare_classes(
    df: pd.DataFrame,
    *,
    label_col: str = "label",
    min_samples_per_class: int = 2,
) -> tuple[pd.DataFrame, List[str]]:
    if min_samples_per_class <= 1:
        return df, []

    counts = df[label_col].value_counts()
    rare = counts[counts < min_samples_per_class].index.tolist()
    if not rare:
        return df, []

    filtered = df[~df[label_col].isin(rare)].reset_index(drop=True)
    return filtered, [str(label) for label in rare]


def _validate_multiclass_split(df: pd.DataFrame, *, test_size: float, label_col: str = "label") -> None:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be a float between 0 and 1 for training/evaluation splits.")

    n_samples = len(df)
    n_classes = int(df[label_col].nunique())
    n_test = math.ceil(n_samples * test_size)
    n_train = n_samples - n_test

    if n_classes < 2:
        raise ValueError("Need at least two classes after filtering.")

    min_count = int(df[label_col].value_counts().min())
    if min_count < 2:
        raise ValueError(
            "Every class needs at least two samples for a stratified split. "
            "Increase min_samples_per_class to drop rare classes."
        )

    if n_test < n_classes:
        min_test_size = n_classes / n_samples
        raise ValueError(
            "test_size is too small for a stratified multiclass split: "
            f"{n_classes} classes require at least {n_classes} test samples. "
            f"Use test_size >= {min_test_size:.3f}."
        )

    if n_train < n_classes:
        max_test_size = (n_samples - n_classes) / n_samples
        raise ValueError(
            "test_size is too large for a stratified multiclass split: "
            f"{n_classes} classes require at least {n_classes} train samples. "
            f"Use test_size <= {max_test_size:.3f}."
        )


def _encode_labels(labels: List[str], label_order: List[str]) -> np.ndarray:
    label_to_id = {label: idx for idx, label in enumerate(label_order)}
    return np.array([label_to_id[label] for label in labels], dtype=np.int64)


def _align_proba_columns(proba: np.ndarray, classes_seen: np.ndarray, n_classes: int) -> np.ndarray:
    aligned = np.zeros((proba.shape[0], n_classes), dtype=np.float64)
    for col_idx, class_id in enumerate(classes_seen):
        aligned[:, int(class_id)] = proba[:, col_idx]
    return aligned


def _eval_multiclass(y_true: np.ndarray, y_pred: np.ndarray, labels: List[int]) -> Dict[str, object]:
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    p_cls, r_cls, f1_cls, support_cls = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(p_weighted),
        "recall_weighted": float(r_weighted),
        "f1_weighted": float(f1_weighted),
        "per_class": {
            str(lbl): {
                "precision": float(p_cls[idx]),
                "recall": float(r_cls[idx]),
                "f1": float(f1_cls[idx]),
                "support": int(support_cls[idx]),
            }
            for idx, lbl in enumerate(labels)
        },
        "confusion_matrix": cm.tolist(),
    }


def train_model_skill_specific(
    data_path: str,
    text_col: str,
    label_col: str,
    split_col: str,
    id_col: str,
    model_name: str,
    model_type: str = "logistic_regression",
    seed: int = 42,
    batch_size: int = 64,
    C: float = 1.0,
    class_weight: str | None = "balanced",
    calibrate: str = "none",
    cache_dir: str | None = "../models/embeddings_cache",
    embedding_task: str | None = None,
    min_samples_per_class: int = 2,
) -> Dict[Any, Dict[str, Any]]:
    """
    Train a classifier for the skill-specific label space iteratively across splits,
    returning item-level predictions and scores mapped by ID.
    """
    logger.info("Loading skill-specific data from %s", data_path)
    # Ensure your read_data_skill_specific is updated to load split_col and id_col
    df = read_data_skill_specific(data_path, text_col, label_col, split_col, id_col)

    df, dropped_rare_classes = _drop_rare_classes(
        df,
        min_samples_per_class=min_samples_per_class,
    )
    if dropped_rare_classes:
        logger.warning(
            "Dropping %d skill-specific class(es) with fewer than %d samples: %s",
            len(dropped_rare_classes),
            min_samples_per_class,
            dropped_rare_classes,
        )

    label_order = sorted(df["label"].unique().tolist())
    logger.info("Loaded %d samples across %d classes", len(df), len(label_order))

    unique_splits = df[split_col].unique()
    predictions_by_id = {}

    for split_val in unique_splits:
        print(f"\n========== Evaluating split: {split_val} ==========")
        
        test_df = df[df[split_col] == split_val]
        train_df = df[df[split_col] != split_val]

        steps = [
            "Split data",
            "Embed train split",
            "Embed test split",
            "Train classifier",
            "Predict on test",
        ]
        progress = tqdm(steps, desc=f"train_model_skill_specific (split={split_val})", unit="step")

        progress.set_description("[1/5] Split data")
        X_train = train_df[text_col].tolist()
        y_train = _encode_labels(train_df[label_col].tolist(), label_order)
        X_test = test_df[text_col].tolist()
        y_test = _encode_labels(test_df[label_col].tolist(), label_order)
        test_ids = test_df[id_col].tolist()
        
        logger.info("Split sizes — train: %d, test: %d", len(X_train), len(X_test))
        progress.update(1)

        progress.set_description("[2/5] Embed train split")
        logger.info("Embedding train split with model '%s' (batch_size=%d)", model_name, batch_size)
        X_train_emb = embed_texts(model_name, X_train, batch_size, cache_dir, task=embedding_task)
        progress.update(1)

        progress.set_description("[3/5] Embed test split")
        logger.info("Embedding test split")
        X_test_emb = embed_texts(model_name, X_test, batch_size, cache_dir, task=embedding_task)
        progress.update(1)

        progress.set_description("[4/5] Train classifier")
        logger.info(
            "Training skill-specific %s - C=%.4f, class_weight=%s, calibrate=%s",
            model_type, C, class_weight, calibrate,
        )
        clf = train_classifier(
            X_train_emb,
            y_train,
            model_type=model_type,
            C=C,
            class_weight=class_weight,
            calibrate=calibrate,
            random_state=seed,
        )
        progress.update(1)

        progress.set_description("[5/5] Predict on test")
        test_proba_raw, _, test_latencies_seconds = predict_proba_with_latency(clf, X_test_emb)
        test_proba = _align_proba_columns(test_proba_raw, clf.classes_, len(label_order))
        test_pred = np.argmax(test_proba, axis=1)

        # Map predictions to the row's unique ID
        for uid, proba, pred_idx, true_idx, lat_seconds in zip(
            test_ids, test_proba, test_pred, y_test, test_latencies_seconds
        ):
            predictions_by_id[uid] = {
                "prediction": label_order[pred_idx],
                "label": label_order[true_idx],
                "scores": {label: float(score) for label, score in zip(label_order, proba)},
                "latency": lat_seconds,
            }
            
        progress.update(1)
        progress.close()

    return predictions_by_id

from typing import Dict, Any

def train_model(data_path: str,
                text_col: str,
                label_col: str,
                split_col: str,
                id_col: str,  # Added parameter to identify rows
                model_name: str,
                model_type: str = "logistic_regression",
                seed: int = 42,
                batch_size: int = 64,
                C: float = 1.0, 
                class_weight: str | None = "balanced",
                calibrate: str = "none",
                cache_dir: str | None = "../../models/embeddings_cache",
                embedding_task: str | None = None
) -> Dict[Any, Dict[str, float | int]]:

    logger.info("Loading data from %s", data_path)
    df = read_data(data_path, text_col, label_col, split_col, id_col)
    logger.info("Loaded %d samples (label distribution: %s)", len(df), df[label_col].value_counts().to_dict())

    unique_splits = df[split_col].unique()
    predictions_by_id = {}
    thr = 0.5  # Standard threshold

    for split_val in unique_splits:
        print(f"\n========== Evaluating split: {split_val} ==========")
        
        test_df = df[df[split_col] == split_val]
        train_df = df[df[split_col] != split_val]

        steps = [
            "Split data",
            "Embed train split",
            "Embed test split",
            "Train classifier",
            "Predict on test",
        ]
        progress = tqdm(steps, desc=f"train_model (split={split_val})", unit="step")

        progress.set_description("[1/5] Split data")
        X_train, y_train = train_df[text_col].tolist(), train_df[label_col].to_numpy()
        X_test, y_test = test_df[text_col].tolist(), test_df[label_col].to_numpy()
        test_ids = test_df[id_col].tolist()
        logger.info("Split sizes — train: %d, test: %d", len(X_train), len(X_test))
        progress.update(1)

        progress.set_description("[2/5] Embed train split")
        logger.info("Embedding train split with model '%s' (batch_size=%d)", model_name, batch_size)
        X_train_emb = embed_texts(model_name, X_train, batch_size, cache_dir, task=embedding_task)
        progress.update(1)

        progress.set_description("[3/5] Embed test split")
        logger.info("Embedding test split")
        X_test_emb = embed_texts(model_name, X_test, batch_size, cache_dir, task=embedding_task)
        progress.update(1)

        progress.set_description("[4/5] Train classifier")
        logger.info("Training %s", model_type)
        clf = train_classifier(X_train_emb, y_train, model_type=model_type, C=C, class_weight=class_weight, calibrate=calibrate, random_state=seed)
        progress.update(1)

        progress.set_description("[5/5] Predict on test")
        test_scores, latency_stats, latency_seconds = predict_proba_positive_with_latency(clf, X_test_emb)
        
        # Map predictions to the row's unique ID
        for uid, score, label, lat_seconds in zip(test_ids, test_scores, y_test, latency_seconds):
            predictions_by_id[uid] = {
                "score": float(score),
                "prediction": int(score >= thr),
                "label": int(label),  # Included so you can calculate metrics against ground truth later
                "latency": lat_seconds,
            }
            
        progress.update(1)
        progress.close()

    return predictions_by_id

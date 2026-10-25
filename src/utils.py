from typing import Dict, Iterable
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve, precision_recall_fscore_support, accuracy_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, average_precision_score
)

_LABEL_MAP_BINARY = {
    "unintended": 0,
    "intended": 1,
}

_LABEL_MAP_SKILL_SPECIFIC = {
    "intended.clear_intent.available_skill.about_me": "about_me",
    "intended.clear_intent.available_skill.active_listener": "active_listener",
    "intended.clear_intent.available_skill.blood_pressure_log": "blood_pressure_log",
    "intended.clear_intent.available_skill.body_weight_log": "body_weight_log",
    "intended.clear_intent.available_skill.calling": "calling",
    "intended.clear_intent.available_skill.change_volume": "change_volume",
    "intended.clear_intent.available_skill.daily_wellbeing": "daily_wellbeing",
    "intended.clear_intent.available_skill.day_planer": "day_planer",
    "intended.clear_intent.available_skill.emergency": "emergency",
    "intended.clear_intent.available_skill.general_reminder": "general_reminder",
    "intended.clear_intent.available_skill.hydro": "hydro",
    "intended.clear_intent.available_skill.knowledge_search": "knowledge_search",
    "intended.clear_intent.available_skill.medication_reminder": "medication_reminder",
    "intended.clear_intent.available_skill.messaging": "messaging",
    "intended.clear_intent.available_skill.mindfulness": "mindfulness",
    "intended.clear_intent.available_skill.mobilisation": "mobilisation",
    "intended.clear_intent.available_skill.news_search": "news_search",
    "intended.clear_intent.available_skill.notebook": "notebook",
    "intended.clear_intent.available_skill.repeat": "repeat",
    "intended.clear_intent.available_skill.sleep_log": "sleep_log",
    "intended.clear_intent.available_skill.standing_test": "standing_test",
    "intended.clear_intent.available_skill.step_counter": "step_counter",
    "intended.clear_intent.available_skill.suicide_risk": "suicide_risk",
    "intended.clear_intent.available_skill.tell_time": "tell_time",
    "intended.clear_intent.available_skill.time_reminder": "time_reminder",
    "intended.clear_intent.available_skill.vital_measure": "vital_measure",
    "intended.clear_intent.available_skill.weather": "weather",
    "intended.clear_intent.available_skill.word_riddle": "word_riddle",
    "intended.clear_intent.available_skill.writing": "writing",
    "intended.clear_intent.unavailable_skill": "out_of_scope",
    "intended.unclear_intent.unspecific": "unclear",
    "intended.unclear_intent.multiple": "unclear",
}

def save_skill_predictions_to_csv(predictions_by_id: dict, output_csv: str) -> pd.DataFrame:
    rows = []
    for uid, data in predictions_by_id.items():
        predicted_class = data["prediction"]
        # Extract the confidence score specifically for the predicted class
        predicted_score = data["scores"].get(predicted_class, 0.0)
        
        rows.append({
            "id": uid,
            "prediction": predicted_class,
            "label": data["label"],
            "score": predicted_score,
            "latency": data["latency"].item()
        })
        
    df_preds = pd.DataFrame(rows)
    df_preds.set_index("id", inplace=True)
    df_preds.to_csv(output_csv)
    return df_preds

def save_skill_llm_predictions_to_csv(predictions_by_id: dict, output_csv: str) -> pd.DataFrame:
    rows = []
    for uid, data in predictions_by_id.items():
        rows.append({
            "id": uid,
            "prediction": data["prediction"],
            "label": data["label"],
            "latency": data["llm_latency_seconds"],
            "invalid_prediction": data["invalid_prediction"],
        })
        
    df_preds = pd.DataFrame(rows)
    df_preds.set_index("id", inplace=True)
    df_preds.to_csv(output_csv)
    return df_preds

def evaluate_predictions(base_file, predictions_file):
    
    def calculate_metrics(group: pd.DataFrame) -> pd.Series:
        y_true = group["label"]
        y_pred = group["prediction"]
        y_score = group["score"]
        
        # Handle edge cases where a split might only have one class
        has_both_classes = len(y_true.unique()) > 1

        y_pred[(y_pred != 1) & (y_pred != 0)] = 1 - y_true[y_pred > 1] #set all out of scope predictions to the false label
        
        return pd.Series({
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, y_score) if has_both_classes else float("nan"),
            "avg_precision": average_precision_score(y_true, y_score) if has_both_classes else float("nan"),
            "latency_mean": group["latency"].mean(),
        })
    
    # Assuming the dataframe from the previous merge is named 'df' 
    # and the column containing fold IDs is named 'split'

    df = pd.read_excel(base_file)
    preds_df = pd.read_csv(predictions_file, index_col=0)
    
    df_merged = df[["cv_fold", "id"]].merge(
        preds_df[["score", "prediction", "label", "latency"]], 
        left_on="id", 
        right_index=True, 
        how="left"
    )
    
    split_metrics = df_merged.groupby("cv_fold").apply(calculate_metrics).reset_index()
    
    # Calculate the mean of the metrics across all folds
    mean_metrics = split_metrics.drop(columns=["cv_fold"]).mean()
    
    # Optional: Calculate the standard deviation to check variance between folds
    std_metrics = split_metrics.drop(columns=["cv_fold"]).std()
    
    print("--- Mean of Fold Means ---")
    print(mean_metrics)
    
    print("\n--- Standard Deviation Across Folds ---")
    print(std_metrics)
    
    return split_metrics

def evaluate_skill_predictions(base_file: str, predictions_file: str):
    
    def calculate_metrics(group: pd.DataFrame) -> pd.Series:
        y_true = group["label"]
        y_pred = group["prediction"]
        
        return pd.Series({
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "latency_mean": group["latency"].mean(),
        })
    
    # Load data
    df = pd.read_excel(base_file) # Change to read_csv if your base file is a CSV
    preds_df = pd.read_csv(predictions_file, index_col=0)
    
    # Merge on 'id'
    df_merged = df[["cv_fold", "id"]].merge(
        preds_df[["prediction", "label", "latency"]], 
        left_on="id", 
        right_index=True, 
        how="inner", #not all ids from the original df are included since only clear intends are considered
    )
    
    # Calculate metrics per fold
    split_metrics = df_merged.groupby("cv_fold").apply(calculate_metrics).reset_index()
    
    # Calculate aggregated statistics
    mean_metrics = split_metrics.drop(columns=["cv_fold"]).mean()
    std_metrics = split_metrics.drop(columns=["cv_fold"]).std()
    
    print("--- Mean of Fold Means ---")
    print(mean_metrics)
    
    print("\n--- Standard Deviation Across Folds ---")
    print(std_metrics)
    
    return split_metrics

def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def read_data(path: str, text_col: str, label_col: str, split_col : str, id_col : str) -> pd.DataFrame:
    """"Read a CSV or Excel file and return a cleaned DataFrame with 'text' and 'label' columns."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
    elif p.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(p, sheet_name="Data")
    else:
        raise ValueError("Unsupported format. Use .csv or .xlsx")

    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Missing required columns. Need '{text_col}' and '{label_col}'")

    df = df[[text_col, label_col, split_col, id_col]]
    df[text_col] = df[text_col].astype(str)
    df[split_col] = df[split_col].astype(int)
    df[label_col] = df[label_col].apply(lambda x: 0 if x == "unintended" else 1)
    df = df.dropna().reset_index(drop=True)
    return df


def read_data_skill_specific(path: str, text_col: str, label_col: str, split_col : str, id_col : str) -> pd.DataFrame:
    """Read a CSV or Excel file, keep only intended utterances, and return a DataFrame
    with 'text' and 'label' columns where label is the specific skill name (e.g. 'weather',
    'change_volume'), 'out_of_scope', or 'unclear'.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p, sep=None, engine="python")
    elif p.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(p, sheet_name="Data")
    else:
        raise ValueError("Unsupported format. Use .csv or .xlsx")

    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Missing required columns. Need '{text_col}' and '{label_col}'")

    df = df[[text_col, label_col, split_col, id_col]]#.rename(columns={text_col: "text", label_col: "label"})
    df[text_col] = df[text_col].astype(str)
    df[label_col] = df[label_col].astype(str)

    # Keep only intended utterances
    df = df[df[label_col].str.startswith("intended.")].copy()

    def map_label(raw: str) -> str:
        if raw in _LABEL_MAP_SKILL_SPECIFIC:
            return _LABEL_MAP_SKILL_SPECIFIC[raw]
        return None  # unexpected: drop

    df[label_col] = df[label_col].apply(map_label)
    df = df.dropna(subset=[label_col]).reset_index(drop=True)
    return df


def eval_at_threshold(y_true: np.ndarray, y_score: np.ndarray, thr: float) -> Dict[str, float]:
    """Evaluate binary classification metrics at a given threshold."""
    y_pred = (y_score >= thr).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = tn / max(1, (tn + fp))
    return {
        "accuracy": float(acc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "specificity": float(spec),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }


def pick_threshold_precision(y_true: np.ndarray, y_score: np.ndarray, precision_target: float):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    # align: thresholds length = len(precisions)-1
    precisions, recalls = precisions[1:], recalls[1:]
    ok = precisions >= precision_target
    if not np.any(ok):
        return 1.0, 0.0, 0.0
    idx = np.argmax(recalls * ok)
    return float(thresholds[idx]), float(precisions[idx]), float(recalls[idx])


def summarize_latency(
    latencies_seconds: Iterable[float | int | None],
    measurement: str = "per_sample",
) -> Dict[str, object]:
    """Summarize observed prediction latencies in seconds."""
    values = []
    for value in latencies_seconds:
        if value is None:
            continue
        try:
            latency = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(latency):
            values.append(latency)

    if not values:
        return {
            "unit": "seconds",
            "measurement": measurement,
            "count": 0,
            "total_seconds": None,
            "mean_seconds": None,
            "median_seconds": None,
            "p90_seconds": None,
            "p95_seconds": None,
            "p99_seconds": None,
            "min_seconds": None,
            "max_seconds": None,
            "std_seconds": None,
            "throughput_per_second": None,
        }

    arr = np.asarray(values, dtype=np.float64)
    total = float(arr.sum())
    return {
        "unit": "seconds",
        "measurement": measurement,
        "count": int(arr.size),
        "total_seconds": total,
        "mean_seconds": float(arr.mean()),
        "median_seconds": float(np.median(arr)),
        "p90_seconds": float(np.percentile(arr, 90)),
        "p95_seconds": float(np.percentile(arr, 95)),
        "p99_seconds": float(np.percentile(arr, 99)),
        "min_seconds": float(arr.min()),
        "max_seconds": float(arr.max()),
        "std_seconds": float(arr.std(ddof=0)),
        "throughput_per_second": float(arr.size / total) if total > 0 else None,
    }


def summarize_batch_latency(
    total_seconds: float,
    count: int,
    measurement: str = "batch",
) -> Dict[str, object]:
    """Summarize a vectorized prediction call where only total batch time is observed."""
    total = float(total_seconds)
    if count <= 0 or not np.isfinite(total):
        return summarize_latency([], measurement=measurement)

    return {
        "unit": "seconds",
        "measurement": measurement,
        "count": int(count),
        "total_seconds": total,
        "mean_seconds": float(total / count),
        "median_seconds": None,
        "p90_seconds": None,
        "p95_seconds": None,
        "p99_seconds": None,
        "min_seconds": None,
        "max_seconds": None,
        "std_seconds": None,
        "throughput_per_second": float(count / total) if total > 0 else None,
    }

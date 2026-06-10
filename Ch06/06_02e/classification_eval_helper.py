import inspect
import json
import os
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

PROJECT_ROOT = Path.cwd().resolve()
load_dotenv(PROJECT_ROOT / '.env')
api_key = os.getenv('GEMINI_API_KEY')
client = genai.Client(api_key=api_key) if api_key else None
GEN_CONFIG = {'temperature': 0.0, 'seed': 42}
FLAG_PROMPT = 'You are a senior data scientist auditing a classification model. Given a metrics dict, identify any red flags or anomalies. Common issues: high accuracy but low recall on imbalanced data, precision-recall tradeoff, weak F1, ROC-AUC close to 0.5. Return ONLY valid JSON: a list of objects with keys flag (string), severity (high/medium/low), explanation (string). No markdown fences. No text outside the JSON.'

def compute_classification_metrics(y_true, y_pred, y_proba=None, threshold=None):
    """Compute standard classification metrics and optionally re-score at a custom threshold."""
    y_true = pd.Series(y_true).astype(int)
    used_threshold = threshold if threshold is not None else 0.50

    if threshold is not None and y_proba is not None:
        y_pred_eval = (pd.Series(y_proba) >= threshold).astype(int)
    else:
        y_pred_eval = pd.Series(y_pred).astype(int)

    metrics = {
        "threshold": float(used_threshold),
        "accuracy": float(accuracy_score(y_true, y_pred_eval)),
        "precision": float(precision_score(y_true, y_pred_eval, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred_eval, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred_eval, zero_division=0)),
        "class_ratio": float(y_true.mean()),
        "predicted_positive_rate": float(pd.Series(y_pred_eval).mean()),
        "confusion_matrix": confusion_matrix(y_true, y_pred_eval).tolist(),
    }

    if y_proba is not None and y_true.nunique() > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    else:
        metrics["roc_auc"] = None

    return metrics


def flag_anomalies(metrics):

    content = f"Metrics: {json.dumps(metrics, indent=2)}"
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=content,
        config={"system_instruction": FLAG_PROMPT, **GEN_CONFIG},
    )
    text = (resp.text or "").strip()
    text = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()
    return json.loads(text)

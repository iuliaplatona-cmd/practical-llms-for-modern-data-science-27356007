import importlib
import json
import os
import re
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

load_dotenv()
client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

GEN_CONFIG = {'temperature': 0.0, 'seed': 42}

SELECTION_PROMPT = 'You are a machine learning engineer. Based ONLY on the dataset profile below (not general ML advice), suggest EXACTLY 3 classifiers to try as baselines. You may choose ANY scikit-learn-compatible classifier (sklearn.*, xgboost.XGBClassifier, etc). Pick the three you think best fit THIS profile.\n\nFor each candidate return:\n  - module: full import path, e.g. \'sklearn.ensemble\'\n  - class:  class name,        e.g. \'RandomForestClassifier\'\n  - reason: MUST cite specific field names and values from the profile. If the reason could apply to any dataset, rewrite it.\n  - init_kwargs: a dict of constructor kwargs to use as-is (e.g. random_state=42, class_weight=\'balanced\' when positive_class_ratio is low). Keep this minimal — we are training baselines with defaults, not tuning.\n\nReturn ONLY valid JSON: {"candidates": [{"module": str, "class": str, "reason": str, "init_kwargs": {...}}, ...]}'

def profile_data(X_train, y_train):
    """Build a short profile the LLM can reason about."""
    X_df = pd.DataFrame(X_train)
    pos_ratio = float(pd.Series(y_train).mean())

    # dtype mix — many categoricals favor trees, all-numeric favors linear models
    n_numeric = int(X_df.select_dtypes(include="number").shape[1])
    n_categorical = int(X_df.shape[1] - n_numeric)

    # max absolute skew across numeric features — heavy skew favors trees
    numeric = X_df.select_dtypes(include="number")
    skew_max = float(numeric.skew().abs().max()) if numeric.shape[1] else 0.0

    # near-constant features (hint of sparse/one-hot columns)
    low_var = int((numeric.var() < 0.01).sum()) if numeric.shape[1] else 0

    # rows-to-features ratio — low values favor regularized linear models
    n_to_p_ratio = round(X_df.shape[0] / max(X_df.shape[1], 1), 2)

    # how correlated are the features (a hint of multicollinearity)
    corr = X_df.corr().abs().to_numpy(copy=True)
    np.fill_diagonal(corr, np.nan)
    max_corr = float(np.nanmax(corr))

    # two quick probes — a linear one and a shallow tree one.
    # The gap between them hints at how non-linear the problem is.
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    linear_probe = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    )
    tree_probe = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    linear_auc = float(
        cross_val_score(linear_probe, X_train, y_train, cv=cv, scoring="roc_auc").mean()
    )
    tree_auc = float(
        cross_val_score(tree_probe, X_train, y_train, cv=cv, scoring="roc_auc").mean()
    )

    return {
        "rows": int(X_df.shape[0]),
        "features": int(X_df.shape[1]),
        "n_numeric": n_numeric,
        "n_categorical": n_categorical,
        "n_to_p_ratio": n_to_p_ratio,
        "positive_class_ratio": round(pos_ratio, 3),
        "max_feature_correlation": round(max_corr, 3),
        "skewness_max": round(skew_max, 3),
        "low_variance_features": low_var,
        "logreg_probe_roc_auc": round(linear_auc, 3),
        "tree_probe_roc_auc": round(tree_auc, 3),
    }


def pick_candidates(profile):
    """Ask the LLM for 3 classifiers. Validate by actually importing the class."""
    content = f"Profile: {json.dumps(profile)}"
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=content,
        config={"system_instruction": SELECTION_PROMPT, **GEN_CONFIG},
    )
    text = re.sub(r"```(?:json)?", "", response.text or "").replace("```", "").strip()
    picks = json.loads(text)["candidates"]
    assert len(picks) == 3, f"Expected 3 candidates, got {len(picks)}"
    for p in picks:
        # Validate the class is real by importing it — guardrail against hallucination.
        mod = importlib.import_module(p["module"])
        p["cls"] = getattr(mod, p["class"])
        p.setdefault("init_kwargs", {})
    return picks


def score_candidate(candidate, X_train, y_train):
    """Fit + 3-fold CV on ROC AUC. No tuning."""
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    clf = candidate["cls"](**candidate["init_kwargs"])
    roc_auc = cross_val_score(
        clf, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
    ).mean()
    return {
        "model": f"{candidate['module']}.{candidate['class']}",
        "cls": candidate["cls"],
        "init_kwargs": candidate["init_kwargs"],
        "roc_auc": roc_auc,
    }


def classification_helper(X_train, X_test, y_train, y_test):
    """Run the full flow: profile → LLM picks 3 → CV-score → fit winner → return results."""
    profile = profile_data(X_train, y_train)
    candidates = pick_candidates(profile)

    leaderboard = (
        pd.DataFrame([score_candidate(c, X_train, y_train) for c in candidates])
        .sort_values("roc_auc", ascending=False)
        .reset_index(drop=True)
    )

    winner = leaderboard.iloc[0]
    model = winner["cls"](**winner["init_kwargs"])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = (
        model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    )

    return {
        "model": model,
        "model_name": winner["model"],
        "init_kwargs": winner["init_kwargs"],
        "profile": profile,
        "candidates": candidates,
        "leaderboard": leaderboard,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }

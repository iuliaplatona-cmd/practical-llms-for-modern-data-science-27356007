import importlib
import json
import os
import re

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_score

load_dotenv()
client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

GEN_CONFIG = {'temperature': 0.0, 'seed': 42}

SELECTION_PROMPT = 'You are a machine learning engineer. Based ONLY on the dataset profile below (not general ML advice), suggest EXACTLY 3 regressors to try as baselines. You may choose ANY scikit-learn-compatible regressor (sklearn.*, xgboost.XGBRegressor, etc). Pick the three you think best fit THIS profile.\n\nFor each candidate return:\n  - module: full import path, e.g. \'sklearn.ensemble\'\n  - class:  class name,        e.g. \'RandomForestRegressor\'\n  - reason: MUST cite specific field names and values from the profile. If the reason could apply to any dataset, rewrite it.\n  - init_kwargs: a dict of constructor kwargs to use as-is (e.g. random_state=42). Keep this minimal — we are training baselines with defaults, not tuning.\n\nReturn ONLY valid JSON: {"candidates": [{"module": str, "class": str, "reason": str, "init_kwargs": {...}}, ...]}'

def profile_data(X_train, y_train):
    """Build a short profile the LLM can reason about."""
    import numpy as np

    X_df = pd.DataFrame(X_train)
    y = pd.Series(y_train)

    n_numeric = int(X_df.select_dtypes(include="number").shape[1])
    n_categorical = int(X_df.shape[1] - n_numeric)

    numeric = X_df.select_dtypes(include="number")
    skew_max = float(numeric.skew().abs().max()) if numeric.shape[1] else 0.0
    low_var = int((numeric.var() < 0.01).sum()) if numeric.shape[1] else 0

    n_to_p_ratio = round(X_df.shape[0] / max(X_df.shape[1], 1), 2)

    corr = X_df.corr().abs().to_numpy(copy=True)
    np.fill_diagonal(corr, np.nan)
    max_corr = float(np.nanmax(corr))

    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    linear_probe = LinearRegression()
    tree_probe = RandomForestRegressor(
        n_estimators=100, max_depth=6, random_state=42, n_jobs=-1
    )
    linear_r2 = float(
        cross_val_score(linear_probe, X_train, y_train, cv=cv, scoring="r2").mean()
    )
    tree_r2 = float(
        cross_val_score(tree_probe, X_train, y_train, cv=cv, scoring="r2").mean()
    )

    return {
        "rows": int(X_df.shape[0]),
        "features": int(X_df.shape[1]),
        "n_numeric": n_numeric,
        "n_categorical": n_categorical,
        "n_to_p_ratio": n_to_p_ratio,
        "target_mean": round(float(y.mean()), 3),
        "target_std": round(float(y.std()), 3),
        "target_min": round(float(y.min()), 3),
        "target_max": round(float(y.max()), 3),
        "target_skew": round(float(y.skew()), 3),
        "max_feature_correlation": round(max_corr, 3),
        "skewness_max": round(skew_max, 3),
        "low_variance_features": low_var,
        "linear_probe_r2": round(linear_r2, 3),
        "tree_probe_r2": round(tree_r2, 3),
    }


def pick_candidates(profile):
    """Ask the LLM for 3 regressors. Validate by actually importing the class."""
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
        # Pin random_state so scores are reproducible regardless of what the LLM returned.
        # LinearRegression has no random_state; ignore kwarg errors at fit-time by only setting it
        # when the class signature accepts it.
        import inspect as _inspect

        if "random_state" in _inspect.signature(p["cls"]).parameters:
            p["init_kwargs"]["random_state"] = 42
    return picks


def score_candidate(candidate, X_train, y_train):
    """Fit + 5-fold CV on RMSE. No tuning."""
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    reg = candidate["cls"](**candidate["init_kwargs"])
    rmse = -cross_val_score(
        reg, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1
    ).mean()
    return {
        "model": f"{candidate['module']}.{candidate['class']}",
        "cls": candidate["cls"],
        "init_kwargs": candidate["init_kwargs"],
        "rmse": rmse,
    }


def regression_helper(X_train, X_test, y_train, y_test):
    """Run the full flow: profile → LLM picks 3 → CV-score → fit winner → return results."""
    profile = profile_data(X_train, y_train)
    candidates = pick_candidates(profile)

    leaderboard = (
        pd.DataFrame([score_candidate(c, X_train, y_train) for c in candidates])
        .sort_values("rmse")
        .reset_index(drop=True)
    )

    winner = leaderboard.iloc[0]
    model = winner["cls"](**winner["init_kwargs"])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        "model": model,
        "model_name": winner["model"],
        "init_kwargs": winner["init_kwargs"],
        "profile": profile,
        "candidates": candidates,
        "leaderboard": leaderboard,
        "y_pred": y_pred,
    }

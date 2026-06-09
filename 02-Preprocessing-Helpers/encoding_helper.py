import os
import re
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from google import genai
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

def build_dataframe_profile(frame: pd.DataFrame) -> pd.DataFrame:
    """Build a compact profile of each column.

    Args:
        frame: Input DataFrame. Not modified in place.

    Returns:
        DataFrame with one row per column.
    """
    missing = frame.isna().sum()
    rows = []

    for col in frame.columns:
        series = frame[col]

        # top values with counts
        top_values = series.value_counts(dropna=True).head(5)
        top_values = [(str(v), int(c)) for v, c in top_values.items()]

        rows.append(
            {
                "column": col,
                "dtype": str(frame[col].dtype),
                "missing": int(missing[col]),
                "missing_pct": round(missing[col] / len(frame) * 100, 1),
                "unique": int(frame[col].nunique(dropna=True)),
                "samples": frame[col].dropna().head(10).tolist(),
            }
        )
    return pd.DataFrame(rows)


def profile_to_str(profile_df: pd.DataFrame) -> str:
    """Convert profile DataFrame to a string for the LLM."""
    lines = []
    for _, row in profile_df.iterrows():
        lines.append(
            f"{row.column} | dtype={row.dtype} | missing={row.missing} ({row.missing_pct}%) | unique={row.unique} | samples={row.samples}"
        )
    return "\n".join(lines)


PLAN_PROMPT = 'You are a feature engineering assistant. Given a dataset profile, return a JSON array of encoding decisions. Each item must have exactly three keys: column, strategy, reason. Valid strategies: skip, binary, ordinal, onehot, scale. Rules: identifier or target -> skip; 2 unique values -> binary; ordered categories -> ordinal; nominal any cardinality -> onehot; continuous numeric -> scale; datetime -> skip. Return ONLY a valid JSON array. No explanation. No markdown fences.'

IMPLEMENT_PROMPT = "You are a senior machine learning engineer. Given a JSON encoding plan and a dataset profile, write executable Python code that builds and fits a ColumnTransformer on the existing DataFrame variable `X_train`. STRICT RULES: 1. Use ONLY these objects: ColumnTransformer, OrdinalEncoder, OneHotEncoder, StandardScaler, pd, np. 2. Do NOT use Pipeline, BaseEstimator, TransformerMixin, FunctionTransformer, or any custom classes. 3. Never hardcode category lists. Use OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1). 4. Use OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=15) for onehot. 5. Do not define any functions or classes. 6. Save the fitted ColumnTransformer to `encoder`. 7. No imports. 8. Always set remainder='passthrough' on the ColumnTransformer so that skipped columns are preserved as-is. Return only executable Python code."

def get_encoding_plan(
    frame: pd.DataFrame,
    target_col: str = None,
    column_types: dict = None,
    show_plan: bool = False,
):
    """Ask LLM for a JSON encoding plan. No code, no exec.

    Args:
        frame:        Input DataFrame.
        target_col:   Target column to skip.
        column_types: Manual overrides e.g. {"JobTitle": "frequency"}.
        show_plan:    If True, print the plan.

    Returns:
        list of dicts — [{column, strategy, reason}, ...]
    """
    work = frame.copy()
    if target_col and target_col in work.columns:
        work = work.drop(columns=[target_col])

    profile_str = profile_to_str(build_dataframe_profile(work))

    if column_types:
        overrides_str = "\n".join([f"  {k}: {v}" for k, v in column_types.items()])
        profile_str += f"\n\nColumn type overrides (use these strategies exactly):\n{overrides_str}"

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=profile_str,
        config={"system_instruction": PLAN_PROMPT, "temperature": 0.0},
    )

    text = re.search(
        r"```(?:python|json)?\s*(.*?)```",
        response.text or "",
        flags=re.DOTALL | re.IGNORECASE,
    )
    text = text.group(1).strip() if text else (response.text or "").strip()
    plan = json.loads(text)

    if show_plan:
        print("--- Encoding Plan ---")
        for d in plan:
            print(
                f"  {d['column']:<30} strategy={d['strategy']:<12} reason={d['reason']}"
            )
        print()

    return plan


def build_encoder(plan: list, frame: pd.DataFrame, show_code: bool = False):
    """Take an approved encoding plan, ask LLM to write the ColumnTransformer code, exec it.

    Args:
        plan:      List of encoding decisions from get_encoding_plan.
        frame:     Training DataFrame to fit on.
        show_code: If True, print generated code before executing.

    Returns:
        (encoder, code) — fitted ColumnTransformer and generated code string.
    """
    profile_str = profile_to_str(build_dataframe_profile(frame))
    plan_str = json.dumps(plan, indent=2)

    prompt = (
        f"Dataset profile:\n{profile_str}\n\n"
        f"Encoding plan:\n{plan_str}\n\n"
        "DataFrame is available as `X_train`. Save fitted ColumnTransformer to `encoder`."
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={"system_instruction": IMPLEMENT_PROMPT, "temperature": 0.0},
    )

    text = response.text or ""
    match = re.search(
        r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE
    )
    code = match.group(1).strip() if match else text.strip()

    if show_code:
        print("--- Generated Code ---")
        print(code)
        print("---\n")

    env = {
        "pd": pd,
        "np": np,
        "X_train": frame.copy(),
        "StandardScaler": StandardScaler,
        "OrdinalEncoder": OrdinalEncoder,
        "OneHotEncoder": OneHotEncoder,
        "ColumnTransformer": ColumnTransformer,
    }
    exec(code, env, env)
    encoder = env.get("encoder")
    if encoder is None:
        raise RuntimeError("Encoder code did not assign `encoder`.")
    return encoder, code

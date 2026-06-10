import contextlib
import io
import json
import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


load_dotenv()
_api_key = os.getenv("GEMINI_API_KEY")
_client = genai.Client(api_key=_api_key) if _api_key else None


def generate_text(system_instruction: str, contents: str) -> str:
    """Send a prompt to Gemini and return the plain text response."""
    if _client is None:
        raise RuntimeError("GEMINI_API_KEY is not set. Add it to your environment before using this pipeline.")
    response = _client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config={
            "temperature": 0.0,
            "seed": 42,
            "system_instruction": system_instruction,
        },
    )
    return (response.text or "").strip()


DIAGNOSE_PROMPT = "You are a data quality expert. Given a dataset profile, identify every data quality issue you can find. For each issue state: the column, the problem, and the recommended fix. Consider: missing values, mixed formats, whitespace, inconsistent casing, identifier columns that should not be modified. Never use infer_datetime_format — removed in pandas 2.2. Return a structured list only. No code."

IMPLEMENT_CLEANING_PROMPT = "You are a data-cleaning assistant. Write executable pandas code using the existing DataFrame variable `df`. Use up-to-date pandas 2.x and Python 3.10+ syntax. For numeric columns with low unique counts (rating scales), prefer mode over median for imputation. Never use infer_datetime_format — removed in pandas 2.2. For mixed date formats use pd.to_datetime(df[col], format='mixed'). Do not use inplace=True. Assign results back to columns. No imports. Save the final cleaned DataFrame to `result_df`. Return only executable Python code."

PLAN_PROMPT = "You are a feature engineering assistant. Given a dataset profile, return a JSON array of encoding decisions. Each item must have exactly three keys: column, strategy, reason. Valid strategies: skip, binary, ordinal, onehot, scale. Rules: identifier or target -> skip; 2 unique values -> binary; ordered categories -> ordinal; nominal any cardinality -> onehot; continuous numeric -> scale; datetime -> skip. Return ONLY a valid JSON array. No explanation. No markdown fences."

IMPLEMENT_ENCODER_PROMPT = "You are a senior machine learning engineer. Given a JSON encoding plan and a dataset profile, write executable Python code that builds and fits a ColumnTransformer on the existing DataFrame variable `X_train`. STRICT RULES: 1. Use ONLY these objects: ColumnTransformer, OrdinalEncoder, OneHotEncoder, StandardScaler, pd, np. 2. Do NOT use Pipeline, BaseEstimator, TransformerMixin, FunctionTransformer, or any custom classes. 3. Never hardcode category lists. Use OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1). 4. Use OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=15) for onehot. 5. Do not define any functions or classes. 6. Save the fitted ColumnTransformer to `encoder`. 7. No imports. 8. Always set remainder='passthrough' on the ColumnTransformer so that skipped columns are preserved as-is. Return only executable Python code."


@dataclass
class PreprocessingResult:
    X_train_enc: np.ndarray
    X_test_enc: np.ndarray
    y_train: pd.Series
    y_test: pd.Series
    encoder: ColumnTransformer
    cleaning_plan: str
    encoding_plan: list[dict]
    cleaning_code: str


def build_dataframe_profile(frame: pd.DataFrame) -> pd.DataFrame:
    """Build a compact profile of each column."""
    missing = frame.isna().sum()
    rows = []

    for col in frame.columns:
        series = frame[col]
        top_values = series.value_counts(dropna=True).head(5)
        top_values = [(str(v), int(c)) for v, c in top_values.items()]

        rows.append(
            {
                "column": col,
                "dtype": str(series.dtype),
                "missing": int(missing[col]),
                "missing_pct": round(missing[col] / len(frame) * 100, 1),
                "unique": int(series.nunique(dropna=True)),
                "top_values": top_values,
                "samples": series.dropna().head(10).tolist(),
            }
        )

    return pd.DataFrame(rows)


def profile_to_str(profile_df: pd.DataFrame) -> str:
    """Convert profile DataFrame to a string for the LLM."""
    lines = []
    for _, row in profile_df.iterrows():
        lines.append(
            f"{row['column']} | dtype={row['dtype']} | "
            f"missing={row['missing']} ({row['missing_pct']}%) | "
            f"unique={row['unique']} | "
            f"top_values={row['top_values']} | "
            f"samples={row['samples']}"
        )
    return "\n".join(lines)


def _extract_code(text: str) -> str:
    match = re.search(r"```(?:python|json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text.strip()


def diagnose_data(frame: pd.DataFrame) -> str:
    profile_str = profile_to_str(build_dataframe_profile(frame))
    return generate_text(DIAGNOSE_PROMPT, profile_str)


def generate_cleaning_code(frame: pd.DataFrame, diagnosis: str) -> str:
    profile_str = profile_to_str(build_dataframe_profile(frame))
    text = generate_text(IMPLEMENT_CLEANING_PROMPT, f"{profile_str}\n\nCleaning plan:\n{diagnosis}")
    return _extract_code(text)


def apply_cleaning_code(frame: pd.DataFrame, code: str) -> pd.DataFrame:
    env = {"pd": pd, "df": frame.copy()}
    exec(code, env, env)
    result = env.get("result_df")
    if result is None:
        raise RuntimeError("Cleaning code did not assign `result_df`.")
    return result


def cleaning_helper(frame: pd.DataFrame, show_code: bool = False, show_diagnosis: bool = False):
    diagnosis = diagnose_data(frame)
    if show_diagnosis:
        print("--- Diagnosis ---")
        print(diagnosis)
        print()

    code = generate_cleaning_code(frame, diagnosis)
    if show_code:
        print("--- Generated Code ---")
        print(code)
        print("---\n")

    result_df = apply_cleaning_code(frame, code)
    return result_df, diagnosis, code


def get_encoding_plan(
    frame: pd.DataFrame,
    target_col: str | None = None,
    column_types: dict | None = None,
    show_plan: bool = False,
) -> list[dict]:
    """Ask the LLM for a JSON encoding plan."""
    work = frame.copy()
    if target_col and target_col in work.columns:
        work = work.drop(columns=[target_col])

    profile_str = profile_to_str(build_dataframe_profile(work))
    if column_types:
        overrides_str = "\n".join([f"  {k}: {v}" for k, v in column_types.items()])
        profile_str += f"\n\nColumn type overrides (use these strategies exactly):\n{overrides_str}"

    text = _extract_code(generate_text(PLAN_PROMPT, profile_str))
    plan = json.loads(text)

    if show_plan:
        print("--- Encoding Plan ---")
        for item in plan:
            print(
                f"  {item['column']:<30} strategy={item['strategy']:<12} reason={item['reason']}"
            )
        print()

    return plan


def build_encoder(plan: list[dict], frame: pd.DataFrame, show_code: bool = False):
    """Generate and fit a ColumnTransformer from an approved plan."""
    profile_str = profile_to_str(build_dataframe_profile(frame))
    plan_str = json.dumps(plan, indent=2)
    prompt = (
        f"Dataset profile:\n{profile_str}\n\n"
        f"Encoding plan:\n{plan_str}\n\n"
        "DataFrame is available as `X_train`. Save fitted ColumnTransformer to `encoder`."
    )

    code = _extract_code(generate_text(IMPLEMENT_ENCODER_PROMPT, prompt))
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


def _normalize_classification_target(y: pd.Series, target_col: str) -> pd.Series:
    y_series = pd.Series(y)
    if y_series.dtype.kind in "biufc":
        return y_series.astype(int)

    y_mapped = (
        y_series.astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": 1, "no": 0})
    )
    if y_mapped.isna().any():
        bad_labels = sorted(y_series[y_mapped.isna()].astype(str).unique().tolist())
        raise ValueError(
            f"Classification target '{target_col}' has unsupported labels: {bad_labels}. "
            "Expected numeric 0/1 or string Yes/No."
        )
    return y_mapped.astype(int)


def _split_raw_data(
    df: pd.DataFrame,
    target_col: str,
    task: str,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stratify = df[target_col] if task == "classification" else None
    return train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )


def _clean_train_and_test(
    train_raw: pd.DataFrame, test_raw: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    with contextlib.redirect_stdout(io.StringIO()):
        train_clean, cleaning_diagnosis, cleaning_code = cleaning_helper(train_raw)
    test_clean = apply_cleaning_code(test_raw, cleaning_code)
    return train_clean, test_clean, cleaning_diagnosis, cleaning_code


def _prepare_features_and_target(
    train_clean: pd.DataFrame,
    test_clean: pd.DataFrame,
    target_col: str,
    task: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_train = train_clean.drop(columns=[target_col])
    X_test = test_clean.drop(columns=[target_col])
    y_train = train_clean[target_col]
    y_test = test_clean[target_col]

    datetime_cols = X_train.select_dtypes(include=["datetime64"]).columns.tolist()
    if datetime_cols:
        X_train = X_train.drop(columns=datetime_cols)
        X_test = X_test.drop(columns=datetime_cols)

    if task == "classification":
        y_train = _normalize_classification_target(y_train, target_col)
        y_test = _normalize_classification_target(y_test, target_col)

    return X_train, X_test, y_train, y_test


def preprocessing_pipeline(
    raw_path: str,
    target_col: str,
    task: str = "classification",
    column_types: dict | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> PreprocessingResult:
    """Run the full preprocessing workflow: split -> clean -> encode -> transform."""
    if task not in ("classification", "regression"):
        raise ValueError(f"task must be 'classification' or 'regression', got '{task}'")

    df = pd.read_csv(raw_path)
    print(f"Loaded: {df.shape}")

    train_raw, test_raw = _split_raw_data(df, target_col, task, test_size, random_state)
    print(f"Split raw: train={train_raw.shape} test={test_raw.shape}")

    train_clean, test_clean, cleaning_diagnosis, cleaning_code = _clean_train_and_test(
        train_raw, test_raw
    )
    print(
        f"Cleaned: train={train_clean.shape} missing={train_clean.isnull().sum().sum()} | "
        f"test={test_clean.shape} missing={test_clean.isnull().sum().sum()}"
    )

    X_train, X_test, y_train, y_test = _prepare_features_and_target(
        train_clean, test_clean, target_col, task
    )

    plan = get_encoding_plan(X_train, column_types=column_types)
    encoder, _ = build_encoder(plan, frame=X_train)

    X_train_enc = encoder.transform(X_train)
    X_test_enc = encoder.transform(X_test)
    print(f"Encoded: train={X_train_enc.shape} test={X_test_enc.shape}")

    return PreprocessingResult(
        X_train_enc=X_train_enc,
        X_test_enc=X_test_enc,
        y_train=y_train,
        y_test=y_test,
        encoder=encoder,
        cleaning_plan=cleaning_diagnosis,
        encoding_plan=plan,
        cleaning_code=cleaning_code,
    )

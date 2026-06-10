import os
import re
import pandas as pd
from google import genai
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

DIAGNOSE_PROMPT = 'You are a data quality expert. Given a dataset profile, identify every data quality issue you can find. For each issue state: the column, the problem, and the recommended fix. Consider: missing values, mixed formats, whitespace, inconsistent casing, identifier columns that should not be modified. Never use infer_datetime_format — removed in pandas 2.2. Return a structured list only. No code.'

IMPLEMENT_PROMPT = "You are a data-cleaning assistant. Write executable pandas code using the existing DataFrame variable `df`. Use up-to-date pandas 2.x and Python 3.10+ syntax. For numeric columns with low unique counts (rating scales), prefer mode over median for imputation. Never use infer_datetime_format — removed in pandas 2.2. For mixed date formats use pd.to_datetime(df[col], format='mixed'). Do not use inplace=True. Assign results back to columns. No imports. Save the final cleaned DataFrame to `result_df`. Return only executable Python code."

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
                "dtype": str(series.dtype),
                "missing": int(missing[col]),
                "missing_pct": round(missing[col] / len(frame) * 100, 1),
                "unique": int(series.nunique(dropna=True)),
                "top_values": top_values,
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
            f"top_values={row['top_values']}"
        )

    return "\n".join(lines)


def diagnose_data(frame: pd.DataFrame):
    profile = build_dataframe_profile(frame)
    profile_str = profile_to_str(profile)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=profile_str,
        config={
            "temperature": 0.0,
            "seed": 42,
            "system_instruction": DIAGNOSE_PROMPT,
        },
    )
    return (response.text or "").strip()


def generate_cleaning_code(frame: pd.DataFrame, diagnosis: str):
    profile = build_dataframe_profile(frame)
    profile_str = profile_to_str(profile)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"{profile_str}\n\nCleaning plan:\n{diagnosis}",
        config={
            "temperature": 0.0,
            "seed": 42,
            "system_instruction": IMPLEMENT_PROMPT,
        },
    )

    text = (response.text or "").strip()
    match = re.search(
        r"```(?:python)?\s*(.*?)```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return match.group(1).strip() if match else text


def apply_cleaning_code(frame: pd.DataFrame, code: str):
    env = {"pd": pd, "df": frame.copy()}
    exec(code, env, env)
    result = env.get("result_df")
    if result is None:
        raise RuntimeError("Generated code did not assign `result_df`.")
    return result


def cleaning_helper(
    frame: pd.DataFrame, show_code: bool = False, show_diagnosis: bool = False
):
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

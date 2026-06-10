import os
import re
import pandas as pd
from google import genai
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

EDA_SUMMARY_PROMPT = 'You are an EDA assistant. Write pandas code using the existing DataFrame variable `df`. Use up-to-date pandas 2.x and Python 3.10+ syntax. Avoid deprecated arguments or methods. Focus on descriptive statistics: shape, dtypes, missing values, unique counts, central tendency and spread. Use only valid pandas operations and function names. Do not invent custom aggregation names or shorthand labels inside pandas methods. If you need quartiles or similar statistics, compute them explicitly with valid pandas code such as quantile(). Store the final result in `result_df` as a DataFrame and not multiple variables.Do NOT create a new DataFrame from scratch. Do NOT include import statements. Return ONLY executable Python code, no explanations.'

def eda_summary_helper(question, frame, show_code=False):
    """Ask a question about a DataFrame; get back a DataFrame.

    Args:
        question:      Plain-English EDA question.
        frame:         Input DataFrame. Not modified in place.
        show_code:     If True, print generated code before executing.

    Returns:
        result DataFrame.
    """
    prompt = (
        f"Columns: {list(frame.columns)}"
        f"Dtypes:{frame.dtypes.to_string()}"
        f"Sample (10 rows):{frame.head(10).to_string()}"
        f"Question: {question}"
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={
            "temperature": 0.0,  # Makes the output deterministic
            "seed": 42,
            "system_instruction": EDA_SUMMARY_PROMPT,  # System level instructions that define rules or behavior for the model
        },
    )

    text = (response.text or "").strip()
    match = re.search(
        r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE
    )
    code = match.group(1).strip() if match else text

    if show_code:
        print("--- Generated Code ---")
        print(code)
        print("---")

    # Define a restricted execution environment.
    # The generated code can only access pandas (pd) and the DataFrame (df).
    env = {"pd": pd, "df": frame.copy()}

    # Execute the generated code in the restricted environment.
    exec(code, env, env)

    # The model is instructed to store the final output in `result_df`.
    result = env.get("result_df")
    if result is None:
        raise RuntimeError("Generated code did not assign `result_df`.")
    return result

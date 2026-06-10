import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

TOOLS_SCHEMA_JSON = {'plot_distribution': {'type': 'object', 'properties': {'column': {'type': 'string'}}, 'required': ['column']}, 'plot_bar_counts': {'type': 'object', 'properties': {'column': {'type': 'string'}}, 'required': ['column']}, 'plot_correlation': {'type': 'object', 'properties': {}}, 'plot_boxplot': {'type': 'object', 'properties': {'column': {'type': 'string'}, 'by': {'type': 'string'}}, 'required': ['column']}}
TOOLS_SCHEMA = types.Tool(
    functionDeclarations=[
        types.FunctionDeclaration(
            name='plot_distribution',
            description='Plot histogram with KDE for a numeric column.',
            parametersJsonSchema=TOOLS_SCHEMA_JSON['plot_distribution'],
        ),
        types.FunctionDeclaration(
            name='plot_bar_counts',
            description='Plot value counts as a bar chart for a categorical column.',
            parametersJsonSchema=TOOLS_SCHEMA_JSON['plot_bar_counts'],
        ),
        types.FunctionDeclaration(
            name='plot_correlation',
            description='Plot correlation heatmap for all numeric columns.',
            parametersJsonSchema=TOOLS_SCHEMA_JSON['plot_correlation'],
        ),
        types.FunctionDeclaration(
            name='plot_boxplot',
            description='Plot boxplot of a numeric column, optionally grouped by a categorical column.',
            parametersJsonSchema=TOOLS_SCHEMA_JSON['plot_boxplot'],
        ),
    ]
)

def plot_distribution(df, column, title=None):
    """Plot histogram with KDE for a numeric column."""
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df[column], kde=True, ax=ax, edgecolor="black")
    ax.set_title(title or f"Distribution of {column}")
    plt.tight_layout()
    plt.show()

def plot_bar_counts(df, column, title=None):
    """Plot value counts as a horizontal bar chart."""
    fig, ax = plt.subplots(figsize=(8, 4))
    df[column].value_counts().plot(kind="barh", ax=ax, edgecolor="black")
    ax.set_title(title or f"Value Counts: {column}")
    plt.tight_layout()
    plt.show()

def plot_correlation(df, title="Correlation Matrix"):
    """Plot correlation heatmap for all numeric columns."""
    numeric_df = df.select_dtypes(include="number")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

def plot_boxplot(df, column, by=None, title=None):
    """Plot boxplot, optionally grouped by a categorical column."""
    fig, ax = plt.subplots(figsize=(8, 4))
    if by:
        sns.boxplot(data=df, x=by, y=column, ax=ax)
    else:
        sns.boxplot(data=df, y=column, ax=ax)
    ax.set_title(title or f"Boxplot: {column}" + (f" by {by}" if by else ""))
    plt.tight_layout()
    plt.show()


def eda_visual_helper(question: str, frame: pd.DataFrame):
    """Ask an EDA question and dispatch the appropriate plotting tool."""
    prompt = (
        f"Columns: {list(frame.columns)}\n"
        f"Dtypes:\n{frame.dtypes.to_string()}\n\n"
        f"Sample (10 rows):\n{frame.head(10).to_string()}\n\n"
        f"Question: {question}"
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0, seed=42, tools=[TOOLS_SCHEMA]
        ),
    )

    part = response.candidates[0].content.parts[0]
    tool_name = part.function_call.name
    args = dict(part.function_call.args)

    if tool_name == "plot_distribution":
        plot_distribution(df=frame.copy(), **args)
    elif tool_name == "plot_bar_counts":
        plot_bar_counts(df=frame.copy(), **args)
    elif tool_name == "plot_correlation":
        plot_correlation(df=frame.copy(), **args)
    elif tool_name == "plot_boxplot":
        plot_boxplot(df=frame.copy(), **args)
    else:
        raise ValueError(f"Unsupported tool requested: {tool_name}")
    return {"tool": tool_name, "args": args}

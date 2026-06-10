import inspect
import json
import os
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

PROJECT_ROOT = Path.cwd().resolve()
load_dotenv(PROJECT_ROOT / '.env')
api_key = os.getenv('GEMINI_API_KEY')
client = genai.Client(api_key=api_key) if api_key else None
GEN_CONFIG = {'temperature': 0.0, 'seed': 42}
FLAG_PROMPT = 'You are a senior data scientist auditing a regression model. Given a metrics dict, identify any red flags or anomalies. Common issues: RMSE much larger than target std, R2 close to 0 or negative, suspiciously high R2 suggesting overfitting, large gap between MAE and RMSE indicating outlier sensitivity. Return ONLY valid JSON: a list of objects with keys flag (string), severity (high/medium/low), explanation (string). No markdown fences. No text outside the JSON.'

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

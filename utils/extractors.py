import json
import httpx
from PyPDF2 import PdfReader
import streamlit as st

ANTHROPIC_API_KEY = st.secrets["ANTHROPIC_API_KEY"]
CLAUDE_MODEL = "claude-3-sonnet-20240229"
API_URL = "https://api.anthropic.com/v1/messages"

def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

def run_llm_pipeline(doc_text, taxonomy_dict, prompt_template):
    prompt = prompt_template.format(
        taxonomy=json.dumps(taxonomy_dict, indent=2),
        document_text=doc_text
    )

    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json"
    }

    body = {
        "model": CLAUDE_MODEL,
        "max_tokens": 1024,
        "system": "You are a helpful assistant that extracts structured data from documents.",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = httpx.post(API_URL, headers=headers, json=body, timeout=30)
        result = response.json()["content"][0]["text"]

        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return {"raw_output": result, "error": "Could not parse JSON"}

    except Exception as e:
        return {"error": f"Claude API error: {str(e)}"}

import json
from PyPDF2 import PdfReader
import streamlit as st
import anthropic
import re

client = anthropic.Anthropic()

def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text


def clean_json_output(text):
    # Remove triple backticks and optional language hints like ```json
    cleaned = re.sub(r"```(?:json)?\\n?", "", text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\\n?```", "", cleaned)
    return cleaned.strip()

def run_llm_pipeline(doc_text, taxonomy_dict, prompt_template):
    prompt = prompt_template.format(
        taxonomy=json.dumps(taxonomy_dict, indent=2),
        document_text=doc_text
    )

    try:
        message = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=1000,
            temperature=0.3,
            system="You are a helpful assistant that extracts structured data from documents.",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        )

        result = message.content[0].text
        cleaned_result = clean_json_output(result)

        try:
            return json.loads(cleaned_result)
        except json.JSONDecodeError:
            return {"raw_output": result, "error": "Could not parse cleaned JSON"}

    except Exception as e:
        return {"error": f"Claude API error: {str(e)}"}

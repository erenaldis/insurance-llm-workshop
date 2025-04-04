import openai
import json
from PyPDF2 import PdfReader
import streamlit as st

openai.api_key = st.secrets["OPENAI_API_KEY"]

def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

def run_llm_pipeline(doc_text, taxonomy_dict, prompt_template):
    prompt = prompt_template.format(
        taxonomy=json.dumps(taxonomy_dict, indent=2),
        document_text=doc_text
    )

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    result = response['choices'][0]['message']['content']

    try:
        parsed_result = json.loads(result)
        return parsed_result
    except json.JSONDecodeError:
        return {"raw_output": result, "error": "Could not parse JSON"}
import json
from PyPDF2 import PdfReader
import streamlit as st
import anthropic

client = anthropic.Anthropic()

def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

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

        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return {"raw_output": result, "error": "Could not parse JSON"}

    except Exception as e:
        return {"error": f"Claude API error: {str(e)}"}

import streamlit as st
import json
from utils.extractors import extract_text_from_pdf, run_llm_pipeline

st.set_page_config(page_title="LLM Insurance Claims Extractor", layout="wide")

# --- Tabs ---
tab1, tab2 = st.tabs(["🔍 Extraction Workflow", "🧪 Prompt Debug / Testing"])

# --- Load default taxonomy ---
with open("taxonomies/default_taxonomy.json") as f:
    default_taxonomy = json.load(f)

default_prompt = """
You are an assistant that extracts structured data from insurance claims documents.
Extract the following fields based on this taxonomy:

{taxonomy}

From the document content below:

--- Document Start ---
{document_text}
--- Document End ---

Return only a JSON object in the format of the taxonomy.
"""

# --- TAB 1: Extraction Workflow ---
with tab1:
    st.title("📄 Insurance Claims Extraction")
    
    uploaded_file = st.file_uploader("Upload an insurance claim document (PDF)", type=["pdf"])
    
    st.subheader("🧷 Define Extraction Taxonomy")
    taxonomy_input = st.text_area("Edit the taxonomy as JSON", json.dumps(default_taxonomy, indent=2), height=250)

    st.subheader("⚙️ LLM Prompt Template")
    prompt_template = st.text_area("Edit the prompt template", default_prompt, height=300)

    if st.button("🚀 Extract Information"):
        if uploaded_file is not None:
            try:
                taxonomy = json.loads(taxonomy_input)
                doc_text = extract_text_from_pdf(uploaded_file)

                with st.spinner("Running LLM..."):
                    extracted_json = run_llm_pipeline(doc_text, taxonomy, prompt_template)

                st.subheader("📤 Extracted Information")
                st.json(extracted_json)

                st.download_button("💾 Download JSON", json.dumps(extracted_json, indent=2),
                                   file_name="extracted_claim.json", mime="application/json")

            except Exception as e:
                st.error(f"Error during extraction: {str(e)}")
        else:
            st.warning("Please upload a document first.")

# --- TAB 2: Prompt Testing ---
with tab2:
    st.title("🧪 Prompt Debugging")

    sample_text = st.text_area("Paste sample claim text here", "John Smith filed a claim on March 3, 2023 for damage to his 2019 Toyota Camry following a hailstorm.")
    taxonomy_debug = st.text_area("Taxonomy (JSON)", json.dumps(default_taxonomy, indent=2), height=200)
    prompt_debug = st.text_area("Prompt Template", default_prompt, height=250)

    if st.button("🧠 Run Prompt Test"):
        try:
            taxonomy = json.loads(taxonomy_debug)
            result = run_llm_pipeline(sample_text, taxonomy, prompt_debug)

            st.subheader("🔍 LLM Output")
            st.json(result)
        except Exception as e:
            st.error(f"Error: {str(e)}")

# pip install langchain pypdf streamlit
from langchain_community.document_loaders import PyPDFLoader

def load_pdf_text(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    return "\n".join([page.page_content for page in pages])

import streamlit as st
import tempfile

st.title("PDF to Text Converter")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        text = load_pdf_text(tmp_file.name)
        st.text_area("Extracted Text", text, height=400)
        
        
# to run this
# streamlit run readPDF.py
# pip install langchain PyPDF2 streamlit
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader, PdfWriter
import os

def split_pdf(input_path, output_path, start_page, end_page):
    reader = PdfReader(input_path)
    writer = PdfWriter()

    total_pages = len(reader.pages)
    start_page = max(1, start_page)
    end_page = min(end_page, total_pages)

    for i in range(start_page - 1, end_page):
        writer.add_page(reader.pages[i])

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "wb") as f:
        writer.write(f)

    return output_path

# Streamlit UI
st.title("PDF Splitter")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
start_page = st.number_input("Start Page", min_value=1, value=1)
end_page = st.number_input("End Page", min_value=1, value=1)
save_folder = st.text_input("Enter folder path to save split PDF", value="D:/Documents")

if uploaded_file and start_page <= end_page:
    file_name = uploaded_file.name.replace(".pdf", f"_pages_{start_page}_to_{end_page}.pdf")
    output_path = os.path.join(save_folder, file_name)

    with open("temp_uploaded.pdf", "wb") as f:
        f.write(uploaded_file.read())

    try:
        saved_path = split_pdf("temp_uploaded.pdf", output_path, start_page, end_page)
        st.success(f"Split PDF saved to: {saved_path}")
    except Exception as e:
        st.error(f"Error saving PDF: {e}")
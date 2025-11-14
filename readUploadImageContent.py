# pip install streamlit // install this library to run this
import streamlit as st
from pydantic import BaseModel, Field
import base64
from pathlib import Path
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Define structured output model
class TextExtract(BaseModel):
    title: str = Field(description="The perceived title on the image")
    main_text: str = Field(description="The main text on the file")
    main_text_en: str = Field(description="The main text on the file translated to English")
    objects_in_image: str = Field(description="Any other objects observed in the image")

# Convert image to base64
def convert_base64(image_path: Path) -> str:
    bytes_data = image_path.read_bytes()
    return base64.b64encode(bytes_data).decode("utf-8")

# Load environment variables
load_dotenv()
model = ChatOpenAI(model="gpt-4.1-mini", api_key=os.getenv("OPENAI_API_KEY"))

# Define prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract the text from the provided image, translate it to English, and describe any objects."),
    ("user", [
        {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,{image_data}"},
        }
    ]),
])

chain = prompt | model.with_structured_output(TextExtract)

# Streamlit UI
st.title("Image Text Extractor with LangChain")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image_path = Path(uploaded_file.name)
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    base64_image = convert_base64(image_path)
    with st.spinner("Processing image..."):
        result = chain.invoke({"image_data": base64_image})
    st.success("Extraction Complete")
    st.json(result.dict())
    
# To run this
# streamlit run readUploadImageContent.py
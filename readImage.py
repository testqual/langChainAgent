from pydantic import BaseModel, Field
import base64
import os
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

class TextExtract(BaseModel):
    title: str = Field(description="The perceived title on the image")
    main_text: str = Field(description="The main text on the file")
    main_text_en: str = Field(description="The main text on the file translated to English")
    objects_in_image: str = Field(description="Any other objects observed in the image")


def convert_base64(image_path: Path) -> str:
    bytes_data = image_path.read_bytes()
    return base64.b64encode(bytes_data).decode("utf-8")

load_dotenv()

model = ChatOpenAI(model="gpt-4.1-mini", api_key=os.getenv("OPENAI_API_KEY"))

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

user_input_path = input("Enter the path to the image file: ")
image_path = Path(user_input_path)
# image_path = Path("byd.jpg")
base64_image = convert_base64(image_path)
result = chain.invoke({"image_data": base64_image})

print(result)
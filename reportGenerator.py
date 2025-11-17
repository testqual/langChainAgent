# pip install langchain-classic
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

loader = PyPDFLoader("testpdf.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(docs)

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a professional analyst. Write a concise summary report based on the following documents."),
    ("human", "{context}")
])

llm = ChatOpenAI()
report_chain = create_stuff_documents_chain(llm, prompt)

# Retrieve relevant chunks and generate report
docs = retriever.invoke("Summarize the key findings from this report")
report = report_chain.invoke({"context": docs})
print(report)


# pip install langchain openai faiss-cpu tiktoken
# pip install unstructured
# pip install python-docx
# pip install python-magic-bin
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader

load_dotenv()

# 1. Load your document
loader = TextLoader("Research_on_ClimateChange.txt", encoding="utf-8")
documents = loader.load()
print("Document loaded successfully")

# 2. Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# 3. Create embeddings and store in FAISS
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)

# 4. Setup retriever
retriever = vectorstore.as_retriever()

# 5. Setup LLM
llm = ChatOpenAI(model_name="gpt-4o-mini")

# 6. Q&A Chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
response = qa_chain.run("What are the key findings?")
print("Q&A Response:", response)

# 7. Summarization Chain
summary_prompt = PromptTemplate.from_template(
    "Summarize the following document:\n\n{context}"
)
summary_chain = create_stuff_documents_chain(llm, summary_prompt)
summary = summary_chain.run(docs)
print("Summary:", summary)
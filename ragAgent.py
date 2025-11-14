from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import create_retriever_tool
from langgraph.prebuilt import create_tool_calling_agent

load_dotenv()

embeddings = OpenAIEmbeddings(model = 'text-embedding-3-large')

texts = [
    'Android is a system that runs on phones and tablets.',
    'It was first made by a company called Android Inc., and Google bought it later.',
    'Android uses parts of another system called Linux.',
    'You can get apps for Android from the Google Play Store.',
    'Many phone brands like Samsung and Xiaomi use Android.',
    'Android versions used to be named after sweets like Oreo and Lollipop.',
    'You can change how Android looks and works with things like widgets and launchers.',
    'Android is used on more phones than any other system in the world.'
]

vector_store  = FAISS.from_texts(texts, embedding=embeddings)

print(vector_store.similarity_search('Android is my phone Operating system.', k=2))

retriever = vector_store.as_retriever(search_kwargs={'k' : 3})

retriever_tool = create_retriever_tool(retriever, name='kb_search', description='Search the same product / fruit knowlegde base for information.')

agent = create_agent(
    model = 'gpt-4.1-mini',
    tools = [retriever_tool],
    system_prompt = ["You are a helpful assistant."]
)

result = agent.invoke({
    "messages" : [{"role": "user", "content" : "What he likes?"}]
})

print(result)
print(result["messages"][-1].content)

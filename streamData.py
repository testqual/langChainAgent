import requests
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

model = init_chat_model(
    model='gpt-4.1-mini',
    temperature=0.1
)

conversation = [
        SystemMessage('You are a helpful assistant for questions regarding programing'),
        HumanMessage('What is langchain?'),
        AIMessage('LangChain is an open-source framework that helps developers build applications powered by large language models'),
        HumanMessage('When was it released?'),
]

# response = model.invoke(conversation)

# print(response)
# print(response.content)
for chunk in model.stream('Hello, What is langchain?'):
    print(chunk.text, end='', flush = True)

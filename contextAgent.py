import requests
from dotenv import load_dotenv
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

@dataclass
class Context:
    user_id: str

@dataclass
class ResponseFormat:
    summary: str
    temperature_celsius: float
    temperature_farenheit: float
    humidity: float

@tool('get_weather', description='Return weather information for a given city', return_direct=False)
def get_weather(city: str):
    response = requests.get(f'https://wttr.in/{city}?format=j1')
    return response.json()

@tool('locate_user', description="Look up a user's city based on the context")
def locate_user(runtime: ToolRuntime[Context]):
    match runtime.context.user_id:
        case '1':
            return 'Melbourne'
        case '2':
            return 'London'
        case '3':
            return 'Paris'
        case _:
            return 'Unknown'

model = init_chat_model('gpt-4.1-mini', temperature=0.3)
checkpointer = InMemorySaver()

agent = create_agent(
    model=model,
    tools=[get_weather, locate_user],
    system_prompt="You are a helpful weather assistant",
    context_schema=Context,
    response_format=ResponseFormat,
    checkpointer=checkpointer
)

config = {'configurable': {'thread_id': 1}}
keyword = input("Enter the city you want to check the weather like in: ")

response = agent.invoke(
    {
        'messages': [
            {'role': 'user', 'content': keyword}
        ]
    },
    config=config,
    context=Context(user_id='ABC123')
)

print(response['structured_response'])
print(response['structured_response'].summary)
print(response['structured_response'].temperature_celsius)
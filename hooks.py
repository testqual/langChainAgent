from dotenv import load_dotenv
from time import time
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

class HooksDemo(AgentMiddleware):
    def __init__(self):
        super().__init__()
        self.start_time = 0.0
        
    def before_agent(self, state: AgentState, runtime):
        self.start_time = time()
        print('before_agent triggered')

    def before_model(self, state: AgentState, runtime):
        print('before_model')

    def after_model(self, state: AgentState, runtime):
        print('after_model')

    def after_agent(self, state: AgentState, runtime):
        print('after_agent:', time()- self.start_time)

user_question = input("Ask a question: ")

agent = create_agent('gpt-4.1-mini', middleware=[HooksDemo()])
response = agent.invoke({
    'messages':[
        SystemMessage('You are a helpful assistant'),
        HumanMessage(user_question)
    ]
})

print(response['messages'][-1].content)


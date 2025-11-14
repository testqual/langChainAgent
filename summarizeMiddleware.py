from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.messages import SystemMessage, HumanMessage

load_dotenv()

agent = create_agent(
    model='gpt-4o-mini',
    middleware=[
        SummarizationMiddleware(
             model='gpt-4o-mini',
             max_tokens_before_summary = 4000,
             messages_to_keep = 20,
             summary_prompt = 'Summarize the most important key points that are relevant for the conversation'
        )
    ]
)
user_question = input("Give me the topic that you want me to explain")

response = agent.invoke({
    'messages': [
        SystemMessage('You are a helpful assistant.'),
        HumanMessage(user_question)
    ]
})

print(response['messages'][-1].content)

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, dynamic_prompt
from dataclasses import dataclass

load_dotenv()

# Base prompt
base_prompt = "You are a helpful AI assistant."

# Define context schema
@dataclass
class Context:
    user_role: str

# Dynamic prompt middleware
@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    user_role = request.runtime.context.user_role

    match user_role:
        case 'expert':
            return f'{base_prompt} Provide detailed technical response.'
        case 'beginner':
            return f'{base_prompt} Keep your explanation simple and basic.'
        case 'child':
            return f'{base_prompt} Explain everything as you were explaining to a four-year-old.'
        case _:
            return base_prompt

# Create agent
agent = create_agent(
    model="gpt-4.1-mini",
    middleware=[user_role_prompt],
    context_schema=Context
)

# Prompt user to select role
print("Select your role:")
print("1. Expert")
print("2. Beginner")
print("3. Child")
role_map = {'1': 'expert', '2': 'beginner', '3': 'child'}
selected = input("Enter the number corresponding to your role: ").strip()
user_role = role_map.get(selected, 'beginner')  # Default to 'beginner' if invalid

# Invoke agent with selected role
response = agent.invoke(
    {
        "messages": [{"role": "user", "content": "Explain PCA"}]
    },
    context=Context(user_role=user_role)
)

print(response)
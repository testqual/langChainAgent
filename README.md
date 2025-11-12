This project uses LangChain to build a research assistant that can query Wikipedia and perform calculations using natural language.

## Features
- Wikipedia search tool
- Calculator tool
- Conversational memory
- OpenAI-powered agent

## Pre- request
Python 3.8 or higher 
OpenAI account(To get the API key) 

## Setup
1. Create virtual environment
	python -m venv .venv

    To change the environment (python development environment) go to command prompt -> Ctrl + Shift + P -> type python: Select Interpreter and select .venv

	pip  install python-dotenv

2. Install dependencies:
    pip install langchain langchain-community langchain-openai python-dotenv

3. Run the agent:
    python wikiSearchAgent.py
    python researchAgent.py


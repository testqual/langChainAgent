from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper

wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
# Get keyword from user input
keyword = input("Enter a topic to search on Wikipedia: ")

# Query the tool with the user-provided keyword
result = wikipedia.run(keyword)

# Display the result
print("\nWikipedia Summary:\n")
print(result)

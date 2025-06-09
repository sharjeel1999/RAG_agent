# from langchain_community.tools.tavily_research import TavilySearchResults
from langchain_community.tools import DuckDuckGoSearchRun

class ToolBox:
    def __init__(self, API_KEY: str):
        self.API_KEY = API_KEY
        self.duckduckgo_runner = DuckDuckGoSearchRun()

    def search(self, query: str, num_results: int = 5):
        return self.tavily_search.run(query = query, num_results = num_results)
    

tools = ToolBox(API_KEY = "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")
print(tools.tavily_search)
from langchain_community.tools.tavily_research import TavilySearchResults

class ToolBox:
    def __init__(self, API_KEY: str):
        self.API_KEY = API_KEY
        self.tavily_search = TavilySearchResults(api_key = API_KEY)

    def search(self, query: str, num_results: int = 5):
        return self.tavily_search.run(query = query, num_results = num_results)
    
    
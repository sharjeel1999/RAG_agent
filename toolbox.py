import json

# from langchain_community.tools.tavily_research import TavilySearchResults
from langchain_community.tools import DuckDuckGoSearchRun

from vectore_store import VectorStore

class ToolBox:
    def __init__(self, API_KEY, llm, model: str = "gpt-4o"):
        self.llm = llm
        self.model = model
        self.vector_store = VectorStore(API_KEY = API_KEY)
        self.duckduckgo_runner = DuckDuckGoSearchRun(num_results = 5)

        self.function_mappings = {
            "duckduckgo_search": self.duckduckgo_search,
            "vector_store_retriever": self.vector_store.retriever
        }

    def duckduckgo_search(self, query: str):
        return self.duckduckgo_runner.run(query)
    
    def get_tools(self):
        vector_store_tool_schema = {
            "type": "function",
            "function": {
                "name": "vector_store_retriever",
                "description": "A tool for retrieving information from a vector store in local storage based on a query. Use this when the user asks a question that can be answered by your internal vector store.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The query string to search in the vector store."
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Optional metadata to filter the search results."
                        }
                    },
                    "required": ["query"]
                }
            }
        }

        duckduckgo_tool_schema = {
            "type": "function",
            "function": {
                "name": "duckduckgo_search", # This is the name the LLM will 'call'
                "description": "A powerful tool for general web/online search, current events, and obtaining up-to-date information. Use this when the user asks a question that cannot be answered by your internal tools or requires external, real-time data.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query string to send to DuckDuckGo."
                        }
                    },
                    "required": ["query"]
                }
            }
        }

        tools = [
            vector_store_tool_schema,
            duckduckgo_tool_schema
        ]
        return tools
    
    def get_tool_names(self, prompt: str):
        tools = self.get_tools()
        messages = [{"role": "user", "content": prompt}]

        response = self.llm.chat.completions.create(
            model = self.model,
            messages = messages,
            tools = tools,
            tool_choice = "auto" # This allows the LLM to choose any of the provided tools
        )

        response_message = response.choices[0].message
        tool_calls = getattr(response_message, 'tool_calls', None)

        functions = []
        if tool_calls:
            for tool in tool_calls:
                function_name = tool.function.name
                arguments = json.loads(tool.function.arguments)
                functions.append({
                    "function_name": function_name,
                    "arguments": arguments
                })
        
        return functions
    
    def execute_tools(self, functions):

        combined_results = []
        for function in functions:
            function_name = function['function_name']
            arguments = function['arguments']
            function_result = self.function_mappings[function_name](**arguments)

            if isinstance(function_result, list):
                combined_results.extend(function_result)
            else:
                combined_results.append(function_result)
        
        # results = []
        # for function in functions:
        #     if function["function_name"] == "duckduckgo_search":
        #         query = function["arguments"].get("query", "")
        #         if query:
        #             result = self.duckduckgo_search(query)
        #             results.append({
        #                 "function_name": function["function_name"],
        #                 "result": result
        #             })
        
        return combined_results
    

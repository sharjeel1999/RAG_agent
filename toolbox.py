import json

# from langchain_community.tools.tavily_research import TavilySearchResults
from langchain_community.tools import DuckDuckGoSearchRun

class ToolBox:
    def __init__(self, llm, model: str = "gpt-4o"):
        self.llm = llm
        self.model = model
        self.duckduckgo_runner = DuckDuckGoSearchRun()

        self.function_mappings = {
            "duckduckgo_search": self.duckduckgo_search
        }

    def duckduckgo_search(self, query: str, num_results: int = 5):
        return self.duckduckgo_runner.run(query=query, num_results=num_results)
    
    def get_tools(self):
        duckduckgo_tool_schema = {
            "type": "function",
            "function": {
                "name": "duckduckgo_search", # This is the name the LLM will 'call'
                "description": "A powerful tool for general web search, current events, and obtaining up-to-date information. Use this when the user asks a question that cannot be answered by your internal tools or requires external, real-time data.",
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
    
    
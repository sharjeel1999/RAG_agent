from openai import OpenAI
import json
from langchain.schema import Document

from vectore_store import VectorStore
from toolbox import ToolBox


class Agent:
    def __init__(self, model, API_KEY):
        self.model = model
        self.llm = OpenAI(api_key = API_KEY)
        self.vector_store = VectorStore(API_KEY = API_KEY)#, data_path = "/home/sharjeel/Desktop/repositories/RAG_agent/vector_data")
        self.toolbox = ToolBox(API_KEY, self.llm, self.model)
        
        self.chunk_size = 600
        self.chunk_overlap = 100

    def Ingest(self, file_path, image_paths = None):
        if file_path != None and image_paths == None:
            self.vector_store.ingest_pdf(
                pdf_file = file_path,
                chunk_size = self.chunk_size,
                chunk_overlap = self.chunk_overlap
            )
        elif image_paths != None and file_path != None:
            self.vector_store.ingest_text_images(
                pdf_file = file_path,
                chunk_size = self.chunk_size,
                chunk_overlap = self.chunk_overlap,
                image_dirs = image_paths
            )

    def serialize_tools(self, tools_output):
        serializable_tools_output = []
        for item in tools_output:
            if isinstance(item, Document):
                # Convert Document object to a serializable dictionary
                serializable_tools_output.append({
                    "page_content": item.page_content,
                    "metadata": item.metadata
                })
            # Handle cases where a tool might return a string directly (like DuckDuckGoSearchRun)
            # or a dictionary, or other basic JSON-compatible types.
            elif isinstance(item, (str, int, float, bool, dict, list)) or item is None:
                serializable_tools_output.append(item)
            else:
                # Fallback for unexpected non-serializable types.
                # You might want to log a warning or convert to string representation.
                print(f"Warning: Found an object of type {type(item).__name__} in tools_output that is not directly JSON serializable. Converting to string.")
                serializable_tools_output.append(str(item))
        
        context = json.dumps(serializable_tools_output, indent = 2)
        return context

    def generate_response(self, prompt, use_context = True):

        tools = self.toolbox.get_tool_names(prompt)

        if use_context:
            # if tools:
            tools_output = self.toolbox.execute_tools(tools)
            context = self.serialize_tools(tools_output)#json.dumps(tools_output)
            # else:
                # context = self.vectore_store.retriever(
                #     query = prompt,
                #     # metadata = {"type": "text"}
                # )

            input_text = (
                "Based on the below context, respond with an accurate answer. If you don't find the answer within the context, say I do not know. Don't repeat the question\n\n"
                f"{context}\n\n"
                f"{prompt}"
            )
        else:
            input_text = prompt

        response = self.llm.chat.completions.create(
            model = self.model,
            messages = [
                {"role": "user", "content": input_text},
            ],
            max_tokens = 150,
            temperature = 0
        )

        return response.choices[0].message.content.strip()


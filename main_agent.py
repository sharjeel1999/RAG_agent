from openai import OpenAI
import json

from vectore_store import VectorStore
from toolbox import Toolbox


class Agent:
    def __init__(self, model, API_KEY):
        self.model = model
        self.llm = OpenAI(api_key = API_KEY)
        self.vector_store = VectorStore(API_KEY = API_KEY)
        self.toolbox = Toolbox(self.llm, self.model)
        
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

    def generate_response(self, prompt, use_context = True):

        tools = self.toolbox.get_tool_names(prompt)

        if use_context:
            if tools:
                tools_output = self.toolbox.execute_tools(tools)
                context = json.dumps(tools_output)
            else:
                context = self.vectore_store.retriever(
                    query = prompt,
                    # metadata = {"type": "text"}
                )

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


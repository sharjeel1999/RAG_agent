import re
import os
from PyPDF2 import PdfReader
from PIL import Image
import base64
from io import BytesIO

import chromadb
from chromadb.utils import embedding_functions

from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.vectorstores import Chroma # deprecated
from langchain_chroma import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings # deprecated
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

class VectorStore:
    def __init__(self, API_KEY, captioning_model: str = "gpt-4o", data_path: str = "./data"):
        self.data_path = data_path

        self.captioning_model_name = captioning_model
        self.client = OpenAI(api_key = API_KEY)

        # chroma_client = chromadb.PersistentClient(path = self.data_path)
        # self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name = "all-MiniLM-L6-v2")
        self.embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        # self.collection = chroma_client.get_or_create_collection(name = "products", embedding_function = self.embedding_function)

    def ingest_documents(self, file_path: str = None, image_paths = None, chunk_size: int = 500, chunk_overlap: int = 100):
        """Ingest documents from the specified file."""
        if file_path and not image_paths:
            if file_path.endswith('.pdf'):
                self.prepare_pdf(file_path, chunk_size, chunk_overlap)
            else:
                raise ValueError("Unsupported file format. Only PDF files are supported.")
        
        elif image_paths and file_path:
            self.ingest_text_images(file_path, chunk_size, chunk_overlap, image_paths)



    def prepare_pdf_docs(self, pdf_file: str, chunk_size: int, chunk_overlap: int, category: str = "data"):
        pdf_loader = PyPDFLoader(pdf_file)
        pdf_documents = pdf_loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
            length_function = len,
            is_separator_regex = False,
        )

        text_chunks = []
        for i, doc in enumerate(pdf_documents):
            
            split_docs = text_splitter.split_documents([doc]) #returnslangchain.schema.Document docs
            for j, chunk in enumerate(split_docs):
                
                chunk.metadata.update({
                    "type": "text",
                    "document_name": os.path.basename(pdf_file),
                    "page": chunk.metadata.get("page"),
                    "chunk_index_on_page": j,
                    "category": category,
                })
                text_chunks.append(chunk)

        return text_chunks




    def encode_image_to_base64(self, image_path):
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            buffered = BytesIO()
            img.save(buffered, format = "JPEG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def get_image_caption_openai(self, image_path):
        base64_image = self.encode_image_to_base64(image_path)
        if not base64_image:
            return "Image not found or could not be processed."

        max_tokens = 300

        response = self.client.chat.completions.create(
            model = self.captioning_model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image concisely and comprehensively. If the image contains any text or equations then include those in the description as well."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    ],
                }
            ],
            max_tokens=max_tokens,
        )
        caption = response.choices[0].message.content
        return caption

    def prepare_image_docs(self, image_dir: str, category: str = "data"):
        
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_docs = []

        for image_file in image_files:
            image_path = os.path.join(image_dir, image_file)
            caption = self.get_image_caption_openai(image_path)

            image_doc = Document(
                page_content = caption,
                metadata = {
                    "source": image_path,
                    "type": "image",
                    "document_name": os.path.basename(image_path),
                    "description_generated": True,
                    "captioning_model": self.captioning_model_name,
                    "category": category,
                    "keywords": None,
                }
            )
            image_docs.append(image_doc)

        return image_docs
    
    def save_documents(self, documents):
        if os.path.exists(self.data_path) and len(os.listdir(self.data_path)) > 0:
            print(f"Loading existing vector store from: {self.data_path}")
            
            vectorstore = Chroma(
                persist_directory = self.data_path,
                embedding_function = self.embedding_function
            )
            print(f"Adding {len(documents)} new documents to the existing store...")
            
            vectorstore.add_documents(documents)
            print("New documents added.")
        else:
            print(f"Creating new vector store at: {self.data_path}")
            
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding_function,
                persist_directory=self.data_path
            )
            print("New vector store created.")

        # vectorstore.persist() # depricated, no longer needed
        print("Vector store persisted successfully.")

    def ingest_pdf(self, pdf_file: str, chunk_size: int, chunk_overlap: int):
        text_docs = self.prepare_pdf_docs(pdf_file, chunk_size, chunk_overlap)
        self.save_documents(text_docs)

    def ingest_text_images(self, pdf_file: str, chunk_size: int, chunk_overlap: int, image_dirs, category: str = "data"):
        text_docs = self.prepare_pdf_docs(pdf_file, chunk_size, chunk_overlap)
        image_docs = self.prepare_image_docs(image_dirs, category)
        self.save_documents(text_docs + image_docs)
        
    
    def retriever(self, query, metadata = None):

        vectorstore = Chroma(
            persist_directory = self.data_path,
            embedding_function = self.embedding_function
        )

        # metadata = {"type": "text"}
        
        if metadata:
            res = vectorstore.similarity_search_with_metadata(
                query = query,
                k = 2,
                where = metadata # Filter based on metadata
            )
        else:
            res = vectorstore.similarity_search(query, k = 2)

        return res

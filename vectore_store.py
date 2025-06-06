import re
import chromadb
from chromadb.utils import embedding_functions

from PyPDF2 import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class VectorStore:
    def __init__(self, data_path: str = "./data"):
        self.data_path = data_path

        chroma_client = chromadb.PersistentClient(path = self.data_path)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name = "all-MiniLM-L6-v2")
        self.collection = chroma_client.get_or_create_collection(name = "products", embedding_function = self.embedding_function)

    def ingest_documents(self, file_path: str, chunk_size: int = 500, chunk_overlap: int = 100):
        """Ingest documents from the specified file."""
        if file_path.endswith('.pdf'):
            self.prepare_pdf(file_path, chunk_size, chunk_overlap)
        else:
            raise ValueError("Unsupported file format. Only PDF files are supported.")


    def load_pdf(self, pdf_file: str, chunk_size: int, chunk_overlap: int):
        reader = PdfReader(pdf_file)    
        documents = {}
        for page_no in range(len(reader.pages)):        
            page = reader.pages[page_no]
            text = page.extract_text() 
            text_chunks = self.get_chunks(text, chunk_size, chunk_overlap)
            documents[page_no] = text_chunks

        return documents

    def get_chunks(self, text: str, chunk_size: int, chunk_overlap: int) -> list:

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,      # Max characters in a chunk
            chunk_overlap = chunk_overlap,    # Overlap between chunks to maintain context
            length_function = len,
            is_separator_regex = False,
        )
        chunks = text_splitter.split_documents(text)
        return chunks

    def add_metadata(self, docs, file_name):
        docs_strings = []  
        ids = []  
        metadatas = []  
        id = 0  
            
        for page_no in docs.keys():
            for doc in docs[page_no]:        
                docs_strings.append(doc)                        
                metadatas.append({'page_no': page_no,"file_name": file_name})
                ids.append(id)
                id += 1

            self.collection.add(
                ids=[str(id) for id in ids],  
                documents=docs_strings,  
                metadatas=metadatas,  
            )
    
    def prepare_pdf(self, pdf_file: str, chunk_size: int, chunk_overlap: int):
        """Prepare the PDF file for vector storage."""
        docs = self.load_pdf(pdf_file, chunk_size, chunk_overlap)
        self.add_metadata(docs, pdf_file)
        
    
    def retriever(self, query):
        vector = self.embedding_function([query])
        results = self.collection.query(    
            query_embeddings=vector,
            n_results=5,
            include=["documents"]
        )
        res = " \n".join(str(item) for item in results['documents'][0])
        return res

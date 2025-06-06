import re
import chromadb
from chromadb.utils import embedding_functions

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class VectorStore:
    def __init__(self, data_path: str = "./data"):
        self.data_path = data_path
        self.documents = []

        chroma_client = chromadb.PersistentClient(path = self.data_path)
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name = "all-MiniLM-L6-v2")
        self.collection = chroma_client.get_or_create_collection(name = "products", embedding_function = embedding_function)

    def load_documents(self, file_path: str):
        """Load documents from the specified file."""
        if file_path.endswith('.pdf'):
            self.load_pdf()
        else:
            raise ValueError("Unsupported file format. Only PDF files are supported.")

    def load_pdf(self, PDF_PATH: str):
        loader = PyPDFLoader(PDF_PATH)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,      # Max characters in a chunk
            chunk_overlap=200,    # Overlap between chunks to maintain context
            length_function=len,
            is_separator_regex=False,
        )
        chunks = text_splitter.split_documents(documents)


    def get_documents(self):
        """Return the loaded documents."""
        return self.documents
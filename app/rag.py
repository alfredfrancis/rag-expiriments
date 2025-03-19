import time
from typing import List, Dict
from functools import lru_cache
from langchain_postgres import PGVector
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI
from openai import OpenAI
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import InMemoryByteStore
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache

from app.config import GEMINI_API_KEY, CONNECTION_STRING, COLLECTION_NAME

# Set up cache for LLM calls
set_llm_cache(InMemoryCache())

base_embeddings = GoogleGenerativeAIEmbeddings(
    google_api_key=GEMINI_API_KEY,
    model="models/embedding-001"
)

# Create a byte store for caching embeddings
store = InMemoryByteStore()

# cached embedder that wraps the base embeddings
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    base_embeddings, 
    store, 
    namespace=base_embeddings.model,
    query_embedding_cache = True
)

openai_client = OpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

class RAGApplication:
    def __init__(self):
        self.embeddings = cached_embedder
        
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Use Gemini through OpenAI-compatible API
        self.llm = ChatOpenAI(
            api_key=GEMINI_API_KEY,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            model="gemini-2.0-flash",
            temperature=0.1
        )
        
        self.initialize_vector_store()
        
    def initialize_vector_store(self):
        """Initialize or connect to the PostgreSQL vector store."""
        try:
            self.vector_store = PGVector(
                connection=CONNECTION_STRING,
                embeddings=self.embeddings,
                collection_name=COLLECTION_NAME,
                use_jsonb=True,
            )
            print("Successfully connected to PostgreSQL vector store")
        except Exception as e:
            print(f"Error connecting to PostgreSQL vector store: {e}")
            print("Creating new vector store...")
            self.vector_store = PGVector.from_documents(
                documents=[],  # Start with empty documents
                embedding=self.embeddings,
                collection_name=COLLECTION_NAME,
                connection=CONNECTION_STRING,
            )
    
    def load_file(self, file_path: str, file_type: str, metadata: Dict = None) -> List[Document]:
        """Load a file based on its type."""
        try:
            if file_type == "pdf":
                loader = PyPDFLoader(file_path)
            else: 
                loader = TextLoader(file_path)
                
            documents = loader.load()
            
            if metadata:
                for doc in documents:
                    doc.metadata.update(metadata)
                    
            return documents
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return []
    
    def process_documents(self, documents: List[Document]) -> int:
        """Process documents and store them in the vector store."""
        if not documents:
            return 0
        
        try:
            splits = self.text_splitter.split_documents(documents)
            
            self.vector_store.add_documents(splits)
            return len(splits)
        except Exception as e:
            print(f"Error processing documents: {e}")
            return 0
    
    def setup_retrieval_chain(self, top_k: int = 4):
        """Set up the retrieval chain for answering questions."""

        template = """
        You are a helpful assistant that answers questions based on the provided context.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer the question based on the context provided. If you cannot find the answer in the context, say "I don't have enough information to answer this question." and suggest what might help.
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k}
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        
        return qa_chain
    
    def answer_question(self, question: str, top_k: int = 4) -> Dict:
        """Answer a question using the retrieval chain."""
        start_time = time.time()
        qa_chain = self.setup_retrieval_chain(top_k)
        
        try:
            result = qa_chain({"query": question})
            
            # Calculate query time in milliseconds
            query_time_ms = (time.time() - start_time) * 1000
            
            return {
                "answer": result["result"],
                "sources": [doc.page_content for doc in result["source_documents"]],
                "query_time_ms": query_time_ms
            }
        except Exception as e:
            print(f"Error answering question: {e}")
            return {
                "answer": "An error occurred while processing your question", 
                "sources": [],
                "query_time_ms": 0
            }
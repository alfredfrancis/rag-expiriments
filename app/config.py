import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up API keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

# PostgreSQL connection string
CONNECTION_STRING = os.getenv("PG_CONNECTION_STRING")
if not CONNECTION_STRING:
    raise ValueError("PG_CONNECTION_STRING not found in environment variables")

# Table name for vector store
COLLECTION_NAME = "document_embeddings"
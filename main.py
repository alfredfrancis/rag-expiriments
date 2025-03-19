import os
import uvicorn
import tempfile
import uuid
from typing import Dict
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from app.models import QueryRequest, QueryResponse, DocumentIngestResponse
from app.rag import RAGApplication

# Initialize FastAPI app
app = FastAPI(
    title="RAG API with Gemini",
    description="API for document ingestion and querying using RAG with Google's Gemini API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG application
rag_app = RAGApplication()

# Background task for document processing
def process_uploaded_file(file_path: str, file_type: str, metadata: Dict):
    documents = rag_app.load_file(file_path, file_type, metadata)
    chunks_processed = rag_app.process_documents(documents)
    print(f"Processed {chunks_processed} chunks from {file_path}")
    
    # Clean up temporary file
    try:
        os.remove(file_path)
    except Exception as e:
        print(f"Error removing temporary file {file_path}: {e}")

@app.post("/documents/ingest", response_model=DocumentIngestResponse)
async def ingest_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    document_id: str = Form(None),
    title: str = Form(None),
    author: str = Form(None),
    source: str = Form(None),
):
    """Ingest a document into the RAG system."""
    
    # Generate document ID if not provided
    if not document_id:
        document_id = str(uuid.uuid4())
    
    # Determine file type
    file_extension = file.filename.split(".")[-1].lower()
    supported_types = ["txt", "pdf"]
    
    if file_extension not in supported_types:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Supported types: {', '.join(supported_types)}")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
        temp_file.write(await file.read())
        temp_file_path = temp_file.name
    
    # Prepare metadata
    metadata = {
        "document_id": document_id,
        "title": title or file.filename,
        "author": author,
        "source": source,
        "filename": file.filename
    }
    
    # Process file in background
    background_tasks.add_task(
        process_uploaded_file,
        temp_file_path,
        file_extension,
        metadata
    )
    
    return DocumentIngestResponse(
        message="Document ingestion started",
        document_id=document_id,
        chunks_processed=0  # Actual count will be processed in background
    )

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the RAG system with a question."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    result = rag_app.answer_question(request.query, request.top_k)
    
    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        query_time_ms=result["query_time_ms"]
    )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model": "gemini-2.0-flash"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
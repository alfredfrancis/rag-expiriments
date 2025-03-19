from typing import List, Dict
from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str
    top_k: int = 4

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    query_time_ms: float

class DocumentIngestResponse(BaseModel):
    message: str
    document_id: str
    chunks_processed: int
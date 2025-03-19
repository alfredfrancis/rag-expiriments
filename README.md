# RAG Application with FastAPI, LangChain, OpenAI, and PostgreSQL Vector Storage

This is a Retrieval-Augmented Generation (RAG) application that uses FastAPI, LangChain, OpenAI's embeddings, and PostgreSQL with pgvector for efficient document storage and retrieval.

## Features

- Document ingestion API endpoint for uploading text and PDF files
- Query API endpoint for asking questions about your documents
- Background document processing
- PostgreSQL vector storage for efficient semantic search
- Caching for LLM calls and document embeddings
- Docker containerization for easy deployment

## Prerequisites

- Docker and Docker Compose
- OpenAI API key

## Quick Start

1. **Clone the repository**

2. **Set up environment variables**

   Copy the example environment file and add your OpenAI API key:
   ```bash
   cp .env.example .env
   ```
   
   Edit the `.env` file and add your OpenAI API key.

3. **Build and start the services**

   ```bash
   docker-compose up -d
   ```

   This will:
   - Build the FastAPI application
   - Start a PostgreSQL database with pgvector extension
   - Initialize the database with required extensions and permissions
   - Connect the application to the database

4. **Use the API**

   The API will be available at http://localhost:8000

   - Swagger UI documentation: http://localhost:8000/docs
   - ReDoc documentation: http://localhost:8000/redoc

## API Usage

### Ingest Documents

```bash
curl -X POST "http://localhost:8000/documents/ingest" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/document.pdf" \
  -F "title=Document Title" \
  -F "author=Author Name" \
  -F "source=Source Website"
```

### Query the System

```bash
curl -X POST "http://localhost:8000/query" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the key benefits of RAG systems?", "top_k": 4}'
```

## Development

To run the application locally without Docker:

1. Install PostgreSQL and the pgvector extension
2. Enable the vector extension in your database
3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Update the `.env` file with your local PostgreSQL connection string
5. Run the application:
   ```bash
   uvicorn app:app --reload
   ```

## File Structure

- `app.py`: Main FastAPI application
- `requirements.txt`: Python dependencies
- `Dockerfile`: Docker image definition
- `docker-compose.yml`: Docker Compose configuration
- `init-db.sql`: PostgreSQL initialization script
- `.env`: Environment variables (not committed to git)
- `.env.example`: Example environment variables file

## Notes

- The database data is persisted in a Docker volume named `postgres-data`
- The application restarts automatically unless explicitly stopped
- Health checks ensure the database is ready before starting the application
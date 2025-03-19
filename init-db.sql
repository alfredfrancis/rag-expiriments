-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create schema if it doesn't exist
CREATE SCHEMA IF NOT EXISTS langchain;

-- Basic permissions
GRANT ALL PRIVILEGES ON DATABASE ragdb TO postgres;
GRANT ALL PRIVILEGES ON SCHEMA langchain TO postgres;
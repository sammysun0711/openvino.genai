# RAG Sample Python Client
### 1. Setup Environment
Download and Install Python. [Python 3.11.9](https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe) is tested on Windows.
```bash
python -m venv rag-client
rag-client\Scripts\activate
pip install langchain langchain_community unstructured markdown
```

### 2. Run script with langchain for text loader/split and sent to RAG server

Usage -h
```bat
usage: client_get_chunks_embeddings.py [-h] --docs DOCS [DOCS ...] [--spliter {Character,RecursiveCharacter,Markdown,Chinese}]
                                       [--chunk_size CHUNK_SIZE] [--chunk_overlap CHUNK_OVERLAP] [--host HOST] [--port PORT]

Process documents and send data to server.

options:
  -h, --help                      show this help message and exit
  --docs DOCS [DOCS ...]          List of documents to process (e.g., test_document_README.md)
  --spliter {Character,RecursiveCharacter,Markdown,Chinese}
                                  Chunking method
  --chunk_size CHUNK_SIZE         Chunk size for processing
  --chunk_overlap CHUNK_OVERLAP   Chunk overlap for smoother processing
  --host HOST                     Server host address default="127.0.0.1"
  --port PORT                     Server port number default=7890
  ```
### 3. Example output: 
```bash
python client_get_chunks_embeddings.py --docs test_document_README.md
```
```bash
get chunks from document with langchain's loader and spliter
loader and spliter finished, len(chunks) is:  15
get_chunks completed! Number of chunks: 15
Init client

response.status:  200
Server response: Init embeddings success.
response.status:  200
Server response: Embeddings success
finished connnection
```

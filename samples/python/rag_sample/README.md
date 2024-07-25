# RAG Sample Python Client
### 1. Setup Environment
Download and Install Python. [Python 3.11.9](https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe) is tested on Windows.
```bash
python -m venv rag-client
rag-client\Scripts\activate
pip install langchain langchain_community unstructured markdown
```

### 2. Run script with langchain for text loader/split and sent to RAG server
```python
python client_get_chunks_embeddings.py --docs test_document_README.md
```

### 3. Example output: 
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

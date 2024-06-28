# usage
conda create -n rag-client python=3.10
pip install langchain

python client_get_chunks_embeddings.py --docs test_document_README.md

log:

get chunks from document with langchain's loader and spliter
loader and spliter finished, len(chunks) is:  15
get_chunks completed! Number of chunks: 15
Init client

response.status:  200
Server response: Init embeddings success.
response.status:  200
Server response: Embeddings success
finished connnection
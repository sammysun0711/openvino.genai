from typing import List
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter, MarkdownTextSplitter
from langchain_community.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader, )
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
import re
import json
import http.client
import argparse
import time

class ChineseTextSplitter(CharacterTextSplitter):
    def __init__(self, pdf: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.pdf = pdf

    def split_text(self, text: str) -> List[str]:
        if self.pdf:
            text = re.sub(r"\n{3,}", "\n", text)
            text = text.replace("\n\n", "")
        sent_sep_pattern = re.compile(
            '([﹒﹔﹖﹗．。！？]["’”」』]{0,2}|(?=["‘“「『]{1,2}|$))')
        sent_list = []
        for ele in sent_sep_pattern.split(text):
            if sent_sep_pattern.match(ele) and sent_list:
                sent_list[-1] += ele
            elif ele:
                sent_list.append(ele)
        return sent_list

TEXT_SPLITERS = {
    "Character": CharacterTextSplitter,
    "RecursiveCharacter": RecursiveCharacterTextSplitter,
    "Markdown": MarkdownTextSplitter,
    "Chinese": ChineseTextSplitter,
}

LOADERS = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PDFMinerLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
}


def load_single_document(file_path: str) -> List[Document]:
    """
    helper for loading a single document

    Params:
      file_path: document path
    Returns:
      documents loaded

    """
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADERS:
        loader_class, loader_args = LOADERS[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()

    raise ValueError(f"File does not exist '{ext}'")



def get_chunks(docs, spliter_name, chunk_size, chunk_overlap):
    """
    load and split

    Params:
      doc: orignal documents provided by user
      chunk_size:  size of a single sentence chunk
      chunk_overlap: overlap size between 2 chunks
    """
    
    start_time_all = time.time() 
    start_time_load = time.time()
    
    documents = []
    for doc in docs:
        documents.extend(load_single_document(doc))

    end_time_load = time.time()
    
    text_splitter = TEXT_SPLITERS[spliter_name](
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    start_time_split = time.time()
    texts = text_splitter.split_documents(documents)
    end_time_split = time.time()  
    print("loader and spliter finished, len(chunks) is: ", len(texts))

    page_content_list = []

    for chunk in texts:
        page_content = chunk.page_content
        page_content_list.append(page_content)

    print(f"get_chunks completed! Number of chunks: {len(page_content_list)}")

    chunks_dict = {"data": page_content_list}  # Assuming sending chunks as JSON data
    json_data = json.dumps(chunks_dict)
    
    end_time_all = time.time()
    load_time_used = end_time_load - start_time_load
    print(f"Time used for loading documents: {load_time_used} seconds")
    split_time_used = end_time_split - start_time_split
    print(f"Time used for splitting documents: {split_time_used} seconds")

    all_time_used = end_time_all - start_time_all
    print(f"Total time used for the whole function: {all_time_used} seconds")

    return json_data


def send_data_to_server(host, port, json_data):
  try:
    print("Init client \n")

    conn = http.client.HTTPConnection(host, port, 30)
    headers = {"Content-Type": "application/json"}  
    conn.request("POST", "/db_init")
    # conn.request("POST", "/embeddings_init")

    response = conn.getresponse()
    print("response.status: ", response.status)
    if response.status == 200:
        print(f"Server response: {response.read().decode('utf-8')}")
    else:
        print(f"Error: Server returned status code {response.status}")

    # with open("C:/llm/LG/xiake_genai/openvino.genai/samples/cpp/rag_sample/document_data.json", "r") as f:
    #     data = json.load(f)
    #     # print(type(data))
    #     # print("len is: ", len(data))
    #     json_data = json.dumps(data)

    # conn.request("POST", "/embeddings", json_data, headers=headers)
    conn.request("POST", "/db_store_embeddings", json_data, headers=headers)
    response = conn.getresponse()
    print("response.status: ", response.status)
    if response.status == 200:
        print(f"Server response: {response.read().decode('utf-8')}")
    else:
        print(f"Error: Server returned status code {response.status}")

    # conn.request("POST", "/db_retrieval", json_data, headers=headers)
    # response = conn.getresponse()
    # print("response.status: ", response.status)
    # if response.status == 200:
    #     print(f"Server response: {response.read().decode('utf-8')}")
    # else:
    #     print(f"Error: Server returned status code {response.status}")

  finally:
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Process documents and send data to server.")
    parser.add_argument("--docs", nargs="+", required=True, default="test_document_README.md", help="List of documents to process (e.g., test_document_README.md)")
    parser.add_argument("--spliter", choices=["Character", "RecursiveCharacter", "Markdown", "Chinese"], default="Character", help="Chunking method")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Chunk size for processing")
    parser.add_argument("--chunk_overlap", type=int, default=200, help="Chunk overlap for smoother processing")
    parser.add_argument("--host", default="127.0.0.1", help="Server host address")
    parser.add_argument("--port", type=int, default=7890, help="Server port number")
    args = parser.parse_args()

    print("Start to load and split document to get chunks from document via Langchain")
    # Get document chunks
    json_data = get_chunks(args.docs, args.spliter, args.chunk_size, args.chunk_overlap)
    # embeddings_init and embeddings
    send_data_to_server(args.host, args.port, json_data)    
  
    print("finished connnection")   


if __name__ == "__main__":
  main()



from typing import List
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter, MarkdownTextSplitter
from langchain.document_loaders import (
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

import http.client




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
    documents = []
    for doc in docs:
        documents.extend(load_single_document(doc))

    text_splitter = TEXT_SPLITERS[spliter_name](
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    texts = text_splitter.split_documents(documents)
    return texts


# RAG configs
docs = ["README.md"]
spliter = "Character" # ["Character", "RecursiveCharacter", "Markdown", "Chinese"]
chunk_size = 1000
chunk_overlap = 200
# get chunks
chunks = get_chunks(docs, spliter, chunk_size, chunk_overlap)
print("get_chunks and len(chunks) is: ", len(chunks))

HOST = "localhost"
PORT = 8080

data_to_send = "This is some data to send to the server!"

conn = http.client.HTTPConnection(HOST, PORT)
conn.request("POST", "/", chunks, headers={"Content-Type": "text/plain"})
response = conn.getresponse()

if response.status == 200:
  print(f"Server response: {response.read().decode('utf-8')}")
else:
  print(f"Error: Server returned status code {response.status}")

conn.close()






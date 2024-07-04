# OpenVINO GenAI Serving (OGS)
Model Server reference implementation based on OpenVINO GenAI Package for Edge/Client AI PC Use Case

## Use Case 1: C++ RAG Sample that supports most popular models like LLaMA 2

This example showcases for Retrieval-Augmented Generation based on text-generation Large Language Models (LLMs): `chatglm`, `LLaMA`, `Qwen` and other models with the same signature and bert model for embedding feature extraction. The sample fearures `ov::genai::LLMPipeline` and configures it for the chat scenario. There is also a Jupyter [notebook](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/254-llm-chatbot) which provides an example of LLM-powered RAG in Python.

### Download and convert the model and tokenizers

The `--upgrade-strategy eager` option is needed to ensure `optimum-intel` is upgraded to the latest version.

```sh
python3 -m pip install --upgrade-strategy eager -r ../../requirements.txt
optimum-cli export openvino --trust-remote-code --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 TinyLlama-1.1B-Chat-v1.0
```

### Setup of PostgreSQL, Libpqxx and Pgvector

#### Langchain's document Loader and Spliter
1. `Load`: `document_loaders` is used to load document data.
2. `Split`: `text_splitter` breaks large Documents into smaller chunks. This is useful both for indexing data and for passing it in to a model, since large chunks are harder to search over and won’t in a model’s finite context window.

#### PostgreSQL

Download `postgresql` from [enterprisedb](https://www.enterprisedb.com/downloads/postgres-postgresql-downloads).(postgresql-16.2-1-windows-x64.exe is tested)

Install PostgreSQL with [postgresqltutorial](https://www..com/postgresql-getting-started/install-postgresql/).
Setup of PostgreSQL:
1. Open `pgAdmin 4` from Windows Search Bar.
2. Click Browser(left side) > Servers > Postgre SQL 10.
3. Create the user `postgres` with password `openvino`.
4. Open `SQL Shell` from Windows Search Bar to check this setup. 'Enter' to set Server, Database, Port, Username as default and type 'openvino' for Password.
 
```bat
Server [localhost]: 
Database [postgres]:
Port [5432]:
Username [postgres]:
Password for user postgres:
```
#### [libpqxx](https://github.com/jtv/libpqxx)
'Official' C++ client library (language binding), built on top of C library.(BSD licence)

Update the source code from https://github.com/jtv/libpqxx in deps\libpqxx



The pipeline connects with DB based on Libpqxx.

#### [pgvector](https://github.com/pgvector/pgvector.git)
Open-source vector similarity search for Postgres

For Windows, Ensure C++ support in Visual Studio 2022 is installed, then use nmake to build in Command Prompt for VS 2022(run as Administrator):
```bat
set "PGROOT=C:\Program Files\PostgreSQL\16"
cd %TEMP%
git clone --branch v0.7.2 https://github.com/pgvector/pgvector.git
cd pgvector
nmake /F Makefile.win
nmake /F Makefile.win install
```
Enable the extension (do this once in each database where you want to use it), run `SQL Shell` from Windows Search Bar with
```bat
CREATE EXTENSION vector;
```
Printing `CREATE EXTENSION` shows successful setup of Pgvector.

#### [pgvector-cpp](https://github.com/pgvector/pgvector-cpp)
pgvector support for C++ (supports libpqxx). 
The headers(pqxx.hpp, vector.hpp, halfvec.hpp) are copied into the local folder rag_sample\include.
Our pipeline do the vector similarity search for the chunks embeddings in PostgreSQL, based on pgvector-cpp.

### Install OpenVINO, VS2022 and Build this pipeline

Download [2024.2 release](https://storage.openvinotoolkit.org/repositories/openvino/packages/2024.2/windows/) from OpenVINO™ archives*. This OV built package is for C++ OpenVINO pipeline, no need to build the source code.
Install latest [Visual Studio 2022 Community](https://visualstudio.microsoft.com/downloads/) for the C++ dependencies and LLM C++ pipeline editing.

#### Windows

Extract the zip file in any location and set the environment variables with dragging this `setupvars.bat` in the terminal `Command Prompt`. `setupvars.ps1` is used for terminal `PowerShell`.`<INSTALL_DIR>` below refers to the extraction location.
Run the following CMD in the terminal `Command Prompt`.

```bat
git submodule update --init
<INSTALL_DIR>\setupvars.bat
cd openvino.genai
cmake -S .\ -B .\build\ && cmake --build .\build\ --config Release -j8
cd .\build\samples\cpp\rag_sample\Release
```
Notice:
- Install on Windows: Copy all the DLL files of PostgreSQL, OpenVINO and tbb and openvino-genai into the release folder. The SQL DLL files locate in the installed PostgreSQL path like "C:\Program Files\PostgreSQL\16\bin". 
- If cmake not installed in the terminal `Command Prompt`, please use the terminal `Developer Command Prompt for VS 2022` instead.
- The ov tokenizer in the third party needs several minutes to build. Set 8 for -j option to specify the number of parallel jobs. 
- Once the cmake finishes, check rag_sample_client.exe and rag_sample_server.exe in the relative path `.\build\samples\cpp\rag_sample\Release`. 
- If Cmake completed without errors, but not find exe, please open the `.\build\OpenVINOGenAI.sln` in VS2022, and set the solution configuration as Release instead of Debug, Then build the llm project within VS2022 again.

### Run:
#### Launch RAG Server
`rag_sample_server.exe --llm_model_path TinyLlama-1.1B-Chat-v1.0 --llm_device CPU --embedding_model_path bge-large-zh-v1.5 --embedding_device CPU  --db_connection "user=postgres host=localhost password=openvino port=5432 dbname=postgres"`
```bat
Usage: rag_sample_server.exe [options]

options:
  -h,    --help                        Show this help message and exit
  --llm_model_path         PATH        Directory contains OV LLM model and tokenizers
  --llm_device             STRING      Specify which device used for llm inference
  --embedding_model_path   PATH        Directory contains OV Bert model and tokenizers
  --embedding_device       STRING      Specify which device used for bert inference
  --db_connection          STRING      Specify which user, host, password, port, dbname
  --max_new_tokens         N           Specify max new generated tokens (default: 256)
  --do_sample              BOOL        Specify whether do random sample (default: False)
  --top_k                  N           Specify top-k parameter for sampling (default: 0)
  --top_p                  N           Specify top-p parameter for sampling (default: 0.7)
  --temperature            N           Specify temperature parameter for sampling (default: 0.95)
  --repeat_penalty         N           Specify penalize sequence of tokens (default: 1.0, means no repeat penalty)
  --verbose                BOOL        Display verbose output including config/system/performance info
```
#### Lanuch RAG Client
`rag_sample_client.exe`
```bat
Init client
Init client finished
Usage:  [options]
options:
  help
  init_embeddings
  embeddings
  db_retrieval
  db_retrieval_llm
  embeddings_unload
  llm_init
  llm
  llm_unload
  health_cheak
  exit
```
To enable Unicode characters for Windows cmd open `Region` settings from `Control panel`. `Administrative`->`Change system locale`->`Beta: Use Unicode UTF-8 for worldwide language support`->`OK`. Reboot.

Discrete GPUs (dGPUs) usually provide better performance compared to CPUs. It is recommended to run larger models on a dGPU with 32GB+ RAM. For example, the model meta-llama/Llama-2-13b-chat-hf can benefit from being run on a dGPU. Modify the source code to change the device for inference to the GPU.

See https://github.com/openvinotoolkit/openvino.genai/blob/master/src/README.md#supported-models for the list of supported models.

#### Lanuch python Client
Use python client to send the message of DB init and send the document chunks to DB for embedding and storing.

samples\python\rag_sample\client_get_chunks_embeddings.py

```bat
conda create -n rag-client python=3.10
pip install langchain
cd samples\python\rag_sample\
python client_get_chunks_embeddings.py --docs test_document_README.md

Start to load and split document to get chunks from document via Langchain
loader and spliter finished, len(chunks) is:  15
get_chunks completed! Number of chunks: 15
Init client

response.status:  200
Server response: Init db success.
response.status:  200
Server response: insert success
finished connnection
```

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
  --host HOST                     Server host address
  --port PORT                     Server port number
  ```


https://github.com/sammysun0711/openvino.genai/assets/102195992/5bffd94b-c853-4848-99a4-4f12a948746f


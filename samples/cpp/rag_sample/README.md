# OpenVINO GenAI Server (OGS)
OpenVINO GenAI Model Server reference implementation based on OpenVINO GenAI API for Edge/Client AI PC use case.

# Why use OpenVINO GenAI Server (OGS)? 
- Most **light-weighted** serving solution without docker container;
- Pure **C++ interface** for edge/client application deployment;
- Pure OpenVINO backend with **minimum package size**;
- **Cross Hardware platform** deployment among Intel CPU/iGPU/dGPU/NPU;

Here is OpenVINO GenAI Server Architecure:
![OpenVINO GenAI Server Architecture](https://github.com/user-attachments/assets/faa394cf-4a03-48db-990e-0a44102b787d "OpenVINO GenAI Server Architecture")


## Use Case 1: C++ RAG Sample that supports most popular models like LLaMA 3

This example showcases for Retrieval-Augmented Generation based on text-generation Large Language Models (LLMs): `chatglm`, `LLaMA`, `Qwen` and other models with the same signature and bert model for embedding feature extraction. The sample fearures `ov::genai::LLMPipeline` and configures it for the chat scenario. There is also a Jupyter [notebook](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/254-llm-chatbot) which provides an example of LLM-powered RAG in Python.

### Download and convert the model and tokenizers

The `--upgrade-strategy eager` option is needed to ensure `optimum-intel` is upgraded to the latest version.
Windows:([Python 3.11.9](https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe) is tested)
#### LLM
```bat
python -m venv rag-sample
rag-sample\Scripts\activate
cd openvino.genai\samples\cpp\rag_sample
python3 -m pip install --upgrade-strategy eager -r ../../requirements.txt
set HF_ENDPOINT=https://hf-mirror.com
optimum-cli export openvino --trust-remote-code --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 TinyLlama-1.1B-Chat-v1.0
```
#### Embedding
BGE embedding is a general Embedding Model. The model is pre-trained using RetroMAE and trained on large-scale pair data using contrastive learning.
Here we provide python script `convert_ov_embedding.py` to download light-weight HF model `BAAI/bge-small-zh-v1.5` and generate one static embedding model for all devices.
The script optimizes the BGE embedding model's parameter precision when loading model to NPU device and also contains the accuracy check on NPU.
```bat
rag-sample\Scripts\activate
cd openvino.genai\samples\cpp\rag_sample
pip install langchain langchain_community unstructured markdown sentence_transformers
python convert_ov_embedding.py
```
Notice:
- Please set the environment variable for [hf-mirror](https://hf-mirror.com/) and try more times, if optimum-cli failed to download model from HF with SSLError.
- If you want to deploy embeddings on NPU, please install latest [Intel® NPU Driver on Windows](https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html), Intel® NPU Driver - Windows* 32.0.100.2688 is tested.
### Setup of PostgreSQL and Pgvector

#### PostgreSQL
Three steps to install PostgreSQL on Windows.
1. Download PostgreSQL Installer for Windows
   Download `postgresql 16.3` from [PostgreSQL installers on the EnterpriseDB](https://www.enterprisedb.com/downloads/postgres-postgresql-downloads). postgresql-16.3-2-windows-x64.exe is tested(367MB).
2. Install PostgreSQL using the installer 
   Besides, many Select-Next for default setting, the key steps are to `create password` and `uncheck Stack Builder`. 
   The screenshots of PostgreSQL graphical installation wizard refer to [postgresqltutorial](https://www.postgresqltutorial.com/postgresql-getting-started/install-postgresql/) and [guide from enterprisedb](https://www.enterprisedb.com/docs/supported-open-source/postgresql/installing/windows/). (old version PostgreSQL)
   
   <details>
   <summary>Click to expand the steps for PostgreSQL 16.3</summary>
   <ol style="list-style-type: decimal;">
   <li>Double-click on the installer file (may need to run as Administrator)</li>
   <li>Select Next. The Installation Directory window opens.</li>
   <li>Select Next. Accept the default installation directory, or specify a location.</li>
   <li>Select components: `Uncheck Stack Builder`. Select Next.</li>
   <li>Select Next. Accept the default location. </li>
   <li>Enter the password for the database superuser (postgres). After entering the password, retype for confirmation. Select Next.</li>
   <li>Select Next. Default port number: 5432.</li>
   <li>Select Next. Select the default locale for the PostgreSQL server.</li>
   <li>Select Next. Review the settings.</li>
   <li>Select Next. The wizard informs: "Ready to install".</li>
   <li>The installation may take a few minutes to complete.</li>
   <li>Click Finish. Complete installation.</li>
   </ol>
   </details> 

3. Test with `SQL Shell`  
    <details>
    <summary>Click to expand the logging</summary>
    
    Open `SQL Shell` from Windows Search Bar. 'Enter' to set Server, Database, Port, Username as default and type your Password.
    ```bat
    Server [localhost]:
    Database [postgres]:
    Port [5432]:
    Username [postgres]:
    Password for user postgres:

    psql (16.3)
    Type "help" for help.

    postgres=#
    ```
    </details> 


#### Pgvector
Open-source vector similarity search for Postgres
Two steps for pgvector:
1. Build and install pgvector
   Download and install [Visual Studio 2022 Community](https://visualstudio.microsoft.com/downloads/). The installation must include the Desktop development with C++ workload, and the C++ MFC for latest v143 build tools (x86 & x64) optional component. Refer to [Install C and C++ support in Visual Studio](https://learn.microsoft.com/en-us/cpp/build/vscpp-step-0-installation?view=msvc-170).
   Run `x64 Native Tools Command Prompt for VS 2022` as Administrator, then use nmake to build and install pgvector:
    ```bat
    set "PGROOT=C:\Program Files\PostgreSQL\16"
    cd %TEMP%
    git clone --branch v0.7.3 https://github.com/pgvector/pgvector.git
    cd pgvector
    nmake /F Makefile.win
    nmake /F Makefile.win install
    ```
2. Enable pgvector extension in Postgres: 
   run `SQL Shell` from Windows Search Bar and type `CREATE EXTENSION vector;`.
   (do this once in each database where you want to use it)
    <details>
    <summary>Click to expand the logging of SQL Shell</summary>

    Open `SQL Shell` from Windows Search Bar. 'Enter' to set Server, Database, Port, Username as default and type your Password.
    ```bat
    Server [localhost]:
    Database [postgres]:
    Port [5432]:
    Username [postgres]:
    Password for user postgres:

    psql (16.3)
    Type "help" for help.

    postgres=# CREATE EXTENSION vector;
    CREATE EXTENSION
    postgres=#
    ```
    Printing `CREATE EXTENSION` shows successful setup of Pgvector.
    </details> 


### Setup and Build OpenVINO GenAI
#### Windows
1. Download and Install VS2022, Cmake and Python:
   - VS2022: Install latest [Visual Studio 2022 Community](https://visualstudio.microsoft.com/downloads/) and Install C and C++ support in Visual Studio.
   - Cmake: If Cmake not installed in the terminal `Command Prompt`, please [download](https://cmake.org/download/) and install Cmake or use the terminal `Developer Command Prompt for VS 2022` instead.
   - Python: the source code building of thirdparty/openvino_tokenizers needs Python3. ([Python 3.11.9](https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe) is tested)
2. Download OpenVINO Runtime:
Download [2024.3 rc2](https://storage.openvinotoolkit.org/repositories/openvino/packages/pre-release/2024.3.0rc2/windows/) from *OpenVINO™ archives*. C++ GenAI pipeline will use the OpenVINO Runtime Dynamic-link library(dll) from the downloaded zip file.
3. Build and install OpenVINO GenAI:
Extract the zip file in any location and set the environment variables with dragging this `setupvars.bat` in the terminal `Command Prompt`. (`setupvars.ps1` is used for terminal `PowerShell`).
`<INSTALL_DIR>` below refers to the extraction location.
Run the following CMD in the terminal `Command Prompt`.
    ```bat
    <INSTALL_DIR>\setupvars.bat
    cd openvino.genai
    git submodule update --init
    cmake -S .\ -B .\build\ && cmake --build .\build\ --config Release -j8
    dir .\build\samples\cpp\rag_sample\Release\rag_sample_client.exe .\build\samples\cpp\rag_sample\Release\rag_sample_server.exe
    ```
    Notice:
    - Once the cmake finishes, check rag_sample_client.exe and rag_sample_server.exe in the relative path `.\build\samples\cpp\rag_sample\Release`. 
    - If Cmake completed without errors, but not find exe, please open the `.\build\OpenVINOGenAI.sln` in VS2022, and set the solution configuration as Release instead of Debug, Then build the llm project within VS2022 again.
    - openvino_tokenizers:
     For development, we download source code of openvino_tokenizers in thirdparty folder(`git submodule update --init`) and build the openvino_tokenizers with openvino-genai. it needs several minutes to build. Set 8 for -j option to specify the number of parallel jobs. 
     In the deployment, the rag-sample could download [openvino_genai package](https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/2024.2/windows) which includes all the DLLs of the openvino_tokenizers, openvino_genai and openvino runtime(must be the same specific version).

    Install on Windows: 
    - Use following CMD lines to copy all the DLL files of PostgreSQL, OpenVINO Runtime, TBB and openvino-genai into the release folder. The SQL DLL files locate in the default path, "C:\Program Files\PostgreSQL\16\bin". 
    - <INSTALL_DIR> below refers to the extraction location of *OpenVINO™ archives* for OpenVINO Runtime.
    ```bat
    cd openvino.genai
    xcopy "C:\Program Files\PostgreSQL\16\bin\*.dll" ".\build\samples\cpp\rag_sample\Release" /s /i
    xcopy ".\build\openvino_genai\*.dll" ".\build\samples\cpp\rag_sample\Release" /s /i
    xcopy "<INSTALL_DIR>\runtime\bin\intel64\Release\*.dll" ".\build\samples\cpp\rag_sample\Release" /s /i
    xcopy "<INSTALL_DIR>\runtime\3rdparty\tbb\bin\*.dll" ".\build\samples\cpp\rag_sample\Release" /s /i
    ```
### Usage:
#### Launch RAG Server
Please use the password you set in the PostgreSQL installation wizard.
```bat
cd openvino.genai
.\build\samples\cpp\rag_sample\Release\rag_sample_server.exe --llm_model_path TinyLlama-1.1B-Chat-v1.0 --llm_device CPU --embedding_model_path bge-small-zh-v1.5 --embedding_device CPU  --db_connection "user=postgres host=localhost password=openvino port=5432 dbname=postgres"
Usage: rag_sample_server.exe [options]

options:
  -h,    --help                        Show this help message and exit
  --llm_model_path         PATH        Directory contains OV LLM model and tokenizers
  --llm_device             STRING      Specify which device used for llm inference
  --embedding_model_path   PATH        Directory contains OV Bert model and tokenizers
  --embedding_device       STRING      Specify which device used for bert inference
  --db_connection          STRING      Specify which user, host, password, port, dbname
  --rag_connection         STRING      Specify host:port(default: "127.0.0.1:7890")
  --max_new_tokens         N           Specify max new generated tokens (default: 32)
  --do_sample              BOOL        Specify whether do random sample (default: False)
  --top_k                  N           Specify top-k parameter for sampling (default: 0)
  --top_p                  N           Specify top-p parameter for sampling (default: 0.7)
  --temperature            N           Specify temperature parameter for sampling (default: 0.95)
  --repeat_penalty         N           Specify penalize sequence of tokens (default: 1.0, means no repeat penalty)
  --verbose                BOOL        Display verbose output including config/system/performance info
```

#### Lanuch Python Client
Launch 2nd command line terminal, use python client to send the message of DB init and send the document chunks to DB for embedding and storing.
Please check the setup of python environment with samples\python\rag_sample\README.md

```bat
rag-sample\Scripts\activate
cd openvino.genai\samples\python\rag_sample\
python client_get_chunks_embeddings.py --docs test_document_README.md
```
Output
```bat
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

#### Lanuch RAG Client
Launch 3rd command line terminal
```bat
cd openvino.genai
.\build\samples\cpp\rag_sample\Release\rag_sample_client.exe
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
  We provide the unit test for embedding with json file.
  <details>
  <summary>Click to expand the logging of embedding unit test. </summary>
  
  User need to type the path of existing json file: absolute path or relate path `.\samples\cpp\rag_sample\document_data.json` 
```bat
.\build\samples\cpp\rag_sample\Release\rag_sample_client.exe
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
init_embeddings
Init embeddings success.
embeddings
This is the unit test for embeddings:
Path of test json file:
.\samples\cpp\rag_sample\document_data.json
Succeed to read the json file.
path: .\samples\cpp\rag_sample\document_data.json
Embeddings success
exit
llm_init
Init llm success.
```
Notice: Please type "exit" to continue to test "llm_init" after successful embedding. 

Corresponding server output:
```bat
Init http server
Port 7890 on IP address 127.0.0.1 is free for OGS.
Load embedding model successed
Load tokenizer model successed
Init embedding models successed
get json_file successed
get inputs successed
size of queries: 15
Start Embedding
shape of embedding_results: (15, 1024)
embedding infer successed
```
  </details> 

#### Complete Usage of RAG Sample
Here is a sample video to demonstrate RAG sample use case on client platform.

https://github.com/sammysun0711/openvino.genai/assets/102195992/c596cd86-dc3c-438f-9fa7-d6395951cec5


The video shows the complete process of RAG:
1. C++ RAG Server: Init server
2. Python client: 
   - Init DB
   - Init Embedding
   - Embedding
   - Store embedding output into DB 
3. C++ RAG client:
   - Init LLM
   - DB Documents Retrival + Chat with LLM

Notice:
- To enable Unicode characters for Windows cmd open `Region` settings from `Control panel`. `Administrative`->`Change system locale`->`Beta: Use Unicode UTF-8 for worldwide language support`->`OK`. Reboot.
- We use [cpp-httplib](https://github.com/yhirose/cpp-httplib) for connection. Larger LLM and longer max_new_tokens need more connection time(default 100 second in rag_sample_client.cpp).
- Besides TinyLlama-1.1B-Chat-v1.0, Qwen2-7B-Instruct is also tested.
- See the list of [supported models](https://github.com/openvinotoolkit/openvino.genai/blob/releases/2024/2/src/docs/SUPPORTED_MODELS.md).

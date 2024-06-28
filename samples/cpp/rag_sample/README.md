# C++ rag sample that supports most popular models like LLaMA 2

This example showcases for Retrieval-Augmented Generation based on text-generation Large Language Models (LLMs): `chatglm`, `LLaMA`, `Qwen` and other models with the same signature and bert model for embedding feature extraction. The sample fearures `ov::genai::LLMPipeline` and configures it for the chat scenario. There is also a Jupyter [notebook](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/254-llm-chatbot) which provides an example of LLM-powered RAG in Python.

## Download and convert the model and tokenizers

The `--upgrade-strategy eager` option is needed to ensure `optimum-intel` is upgraded to the latest version.

```sh
python3 -m pip install --upgrade-strategy eager -r ../../requirements.txt
optimum-cli export openvino --trust-remote-code --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 TinyLlama-1.1B-Chat-v1.0
```

## Setup of PostgreSQL and Libpqxx
PostgreSQL

Download postgresql with this link:
https://www.enterprisedb.com/downloads/postgres-postgresql-downloads
(postgresql-16.2-1-windows-x64.exe is tested)

Install PostgreSQL, open pgAdmin 4 and create a user with password with the following guide: 
https://www.runoob.com/postgresql/windows-install-postgresql.html

Libpqxx

The official C++ client API for PostgreSQL.

Update the source code from https://github.com/jtv/libpqxx in deps\libpqxx:
git submodule update --init

Copy all the DLL files into the release folder like the DLL files of OpenVINO and tbb and openvino-genai.
The DLL files locate in the installed PostgreSQL path like "C:\Program Files\PostgreSQL\16\bin"


## Install OpenVINO, VS2022 and Build this pipeline

Download [2024.2 release](https://storage.openvinotoolkit.org/repositories/openvino/packages/2024.2/windows/) from OpenVINO™ archives*. This OV built package is for C++ OpenVINO pipeline, no need to build the source code.
Install latest [Visual Studio 2022 Community](https://visualstudio.microsoft.com/downloads/) for the C++ dependencies and LLM C++ pipeline editing.

### Windows

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
- If cmake not installed in the terminal `Command Prompt`, please use the terminal `Developer Command Prompt for VS 2022` instead.
- The ov tokenizer in the third party needs several minutes to build. Set 8 for -j option to specify the number of parallel jobs. 
- Once the cmake finishes, check rag_sample_client.exe and rag_sample_server.exe in the relative path `.\build\samples\cpp\rag_sample\Release`. 
- If Cmake completed without errors, but not find exe, please open the `.\build\OpenVINOGenAI.sln` in VS2022, and set the solution configuration as Release instead of Debug, Then build the llm project within VS2022 again.

## Run:
### Launch RAG Server
`rag_sample_server --llm_model_path TinyLlama-1.1B-Chat-v1.0 --llm_device CPU`

### Lanuch RAG Client
`rag_sample_client`

To enable Unicode characters for Windows cmd open `Region` settings from `Control panel`. `Administrative`->`Change system locale`->`Beta: Use Unicode UTF-8 for worldwide language support`->`OK`. Reboot.

Discrete GPUs (dGPUs) usually provide better performance compared to CPUs. It is recommended to run larger models on a dGPU with 32GB+ RAM. For example, the model meta-llama/Llama-2-13b-chat-hf can benefit from being run on a dGPU. Modify the source code to change the device for inference to the GPU.

See https://github.com/openvinotoolkit/openvino.genai/blob/master/src/README.md#supported-models for the list of supported models.

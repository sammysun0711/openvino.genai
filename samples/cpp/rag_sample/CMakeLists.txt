# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
# fix httplib issue
if (MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
endif()
# SET(CMAKE_BUILD_TYPE "Debug")
# SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")

find_package(OpenVINOGenAI REQUIRED PATHS
    "${CMAKE_BINARY_DIR}"  # Reuse the package from the build.
    ${OpenVINO_DIR}  # GenAI may be installed alogside OpenVINO.
)

include_directories(include)

find_package(OpenVINO REQUIRED COMPONENTS Runtime)

add_executable(rag_sample_server rag_sample_server.cpp src/handle_master.cpp src/embeddings.cpp src/db_pgvector.cpp src/reranker.cpp)
add_executable(rag_sample_client rag_sample_client.cpp)
add_subdirectory(deps/libpqxx build-pqxx)
# set MT for libpqxx
if (TARGET libpqxx AND MSVC)
  target_compile_options(libpqxx PUBLIC /MT)
  target_link_options(libpqxx PUBLIC /NODEFAULTLIB:MSVCRT)
endif()
target_link_libraries(rag_sample_server PRIVATE pqxx)
target_link_libraries(rag_sample_server PRIVATE openvino::genai)

target_compile_definitions(rag_sample_server PRIVATE OPENVINO_TOKENIZERS_PATH="${OPENVINO_TOKENIZERS_PATH}")
set(OPENVINO_TOKENIZERS_PATH $<IF:$<BOOL:${genai_imported}>,${genai_imported},$<TARGET_FILE_DIR:openvino::genai>>)
set(OPENVINO_TOKENIZERS_FILENAME "${CMAKE_SHARED_LIBRARY_PREFIX}openvino_tokenizers${CMAKE_SHARED_LIBRARY_SUFFIX}")
target_compile_definitions(rag_sample_server PRIVATE
    OPENVINO_TOKENIZERS_PATH="${OPENVINO_TOKENIZERS_PATH}/${OPENVINO_TOKENIZERS_FILENAME}")

target_compile_features(rag_sample_server PRIVATE cxx_std_17)
target_compile_features(rag_sample_client PRIVATE cxx_std_17)

install(TARGETS rag_sample_server rag_sample_client 
    RUNTIME DESTINATION samples_bin/
    COMPONENT samples_bin
    EXCLUDE_FROM_ALL)

cmake_minimum_required(VERSION 3.15)
project(openvino_backend_api VERSION 1.0 LANGUAGES CXX)

#set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib CACHE STRING "")
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib CACHE STRING "")
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin CACHE STRING "")
#add_subdirectory(../../thirdparty/openvino_tokenizers/ "${CMAKE_CURRENT_BINARY_DIR}/openvino_tokenizers/")
set(OPENVINO_TOKENIZERS_PATH ${CMAKE_CURRENT_SOURCE_DIR}/openvino_tokenizers-2024.1.0.0-Linux/libopenvino_tokenizers.so)
#if(TARGET openvino_tokenizers)
#    set(OPENVINO_TOKENIZERS_PATH $<TARGET_FILE:openvino_tokenizers>)
#else()
#    set(OPENVINO_TOKENIZERS_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../bin/openvino_tokenizers.dll)  # TODO: I'll go away after the generate() gets a way to find openvino_tokenizers
#endif()
set(CMAKE_CXX_STANDARD 17)
if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall /wd4710 /wd4711")
    add_definitions("/wd4267 /wd4244 /wd4305 /Zc:strictStrings /utf-8")
    add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/source-charset:utf-8>")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -fPIC")
    set_source_files_properties(${CPP_SOURCES} PROPERTIES COMPILE_FLAGS "-pedantic-errors")
endif ()

find_package(OpenVINO REQUIRED COMPONENTS Runtime)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif ()
#add_subdirectory(../../thirdparty/openvino_tokenizers/ "${CMAKE_CURRENT_BINARY_DIR}/openvino_tokenizers/")

#target_compile_definitions(${TARGET_NAME} PRIVATE OPENVINO_TOKENIZERS_PATH=\"$<TARGET_FILE:openvino_tokenizers>\")
# third-party libraries
set(ABSL_ENABLE_INSTALL ON)
set(ABSL_PROPAGATE_CXX_STD ON)
#add_subdirectory(../../thirdparty/openvino_tokenizers/ "${CMAKE_CURRENT_BINARY_DIR}/openvino_tokenizers/")
add_subdirectory(third_party/abseil-cpp)
add_subdirectory(third_party/re2)

file(GLOB CPP_SOURCES
  ${PROJECT_SOURCE_DIR}/*.h
  ${PROJECT_SOURCE_DIR}/*.hpp
  ${PROJECT_SOURCE_DIR}/*.cpp)

add_library(qwen STATIC qwen.cpp)
add_library(openvino_backend_api SHARED openvino_backend_api.cpp)
target_link_libraries(qwen PUBLIC re2::re2)

add_executable(test test.cpp)
target_link_libraries(test PUBLIC openvino_backend_api)
target_link_libraries(openvino_backend_api PUBLIC openvino::runtime qwen)

add_executable(rag_model_server rag_model_server.cpp)
#target_compile_definitions(rag_model_server PRIVATE OPENVINO_TOKENIZERS_PATH=\"$<TARGET_FILE:openvino_tokenizers>\")
target_compile_definitions(rag_model_server PRIVATE OPENVINO_TOKENIZERS_PATH="${OPENVINO_TOKENIZERS_PATH}")
#target_compile_definitions(rag_model_server PRIVATE OPENVINO_TOKENIZERS_PATH="${OPENVINO_TOKENIZERS_PATH}")

add_executable(rag_model_client rag_model_client.cpp)
#include_directories(cpp-httplib-0.15.3)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(rag_model_server PUBLIC Threads::Threads)
target_link_libraries(rag_model_client PUBLIC Threads::Threads)

target_link_libraries(rag_model_server PRIVATE openvino_backend_api openvino::runtime Threads::Threads)
set_target_properties(rag_model_server PROPERTIES CXX_STANDARD 17)
set_target_properties(rag_model_server PROPERTIES CXX_STANDARD_REQUIRED ON)

target_link_libraries(rag_model_client PRIVATE Threads::Threads)
set_target_properties(rag_model_client PROPERTIES CXX_STANDARD 17)
set_target_properties(rag_model_client PROPERTIES CXX_STANDARD_REQUIRED ON)


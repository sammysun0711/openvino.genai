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

add_executable(rag_sample_server rag_sample_server.cpp src/handle_master.cpp src/embeddings.cpp src/db_pgvector.cpp src/reranker.cpp src/blip.cpp)

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


# gui
set(GUI_APP_NAME "rag_sample_gui")

FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.4
)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.91.0
)
FetchContent_MakeAvailable(imgui)

if(NOT imgui_POPULATED)
    FetchContent_Populate(imgui)
    add_subdirectory(${imgui_SOURCE_DIR} ${imgui_BINARY_DIR})
endif()

message(STATUS "imgui_SOURCE_DIR: ${imgui_SOURCE_DIR}")
message(STATUS "imgui_BINARY_DIR: ${imgui_BINARY_DIR}")

set(IMGUI_SOURCES
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)

set(APP_DEPS_SOURCE
    src/app/tinyfiledialogs.c
)

add_executable(${GUI_APP_NAME} rag_sample_gui.cpp
    src/handle_master.cpp
    src/embeddings.cpp
    src/db_pgvector.cpp
    src/blip.cpp
    src/app/worker.cpp
    src/app/server_context.cpp
    ${APP_DEPS_SOURCE}
    ${IMGUI_SOURCES}
)
target_include_directories(${GUI_APP_NAME} PRIVATE
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
    ${CMAKE_CURRENT_SOURCE_DIR}/include/app
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(${GUI_APP_NAME} PRIVATE
    pqxx
    openvino::genai
    glfw)
find_package(OpenGL REQUIRED)
target_link_libraries(${GUI_APP_NAME} PRIVATE OpenGL::GL)
target_compile_definitions(${GUI_APP_NAME} PRIVATE OPENVINO_TOKENIZERS_PATH="${OPENVINO_TOKENIZERS_PATH}")
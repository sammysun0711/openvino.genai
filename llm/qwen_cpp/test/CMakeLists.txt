cmake_minimum_required(VERSION 3.12)
project(test VERSION 1.0 LANGUAGES CXX)

if (NOT "$ENV{OPENVINO_BACKEND_BUILD_TYPE}" STREQUAL "")
    set(OPENVINO_BACKEND_BUILD_TYPE "$ENV{OPENVINO_BACKEND_BUILD_TYPE}" CACHE INTERNAL "Copied from environment variable")
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib CACHE STRING "")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib CACHE STRING "")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin CACHE STRING "")

set(CMAKE_CXX_STANDARD 17)
find_package(OpenVINO REQUIRED COMPONENTS Runtime)


# third-party libraries
set(ABSL_ENABLE_INSTALL ON)
set(ABSL_PROPAGATE_CXX_STD ON)
add_subdirectory(third_party/abseil-cpp)
add_subdirectory(third_party/re2)


file(GLOB CPP_SOURCES
  ${PROJECT_SOURCE_DIR}/*.h
  ${PROJECT_SOURCE_DIR}/*.hpp
  ${PROJECT_SOURCE_DIR}/*.cpp)

set(OPENVINO_BACKEND_API_LIBRARY_DIRECTORY ../lib/${OPENVINO_BACKEND_BUILD_TYPE})
add_executable(test test.cpp)
target_include_directories(test PUBLIC include)
target_link_libraries(test PUBLIC ${OPENVINO_BACKEND_API_LIBRARY_DIRECTORY}/openvino_backend_api 
                                  ${OPENVINO_BACKEND_API_LIBRARY_DIRECTORY}/qwen 
                                  openvino::runtime 
                                  re2::re2)

if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall /wd4710 /wd4711")
    add_definitions("/wd4267 /wd4244 /wd4305 /Zc:strictStrings /utf-8")
    add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/source-charset:utf-8>")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall")
    set_source_files_properties(${CPP_SOURCES} PROPERTIES COMPILE_FLAGS "-pedantic-errors")
endif ()
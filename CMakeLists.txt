cmake_minimum_required(VERSION 3.16)
project(qenv CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)   # needed to link Quackle into a Python module
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Where the Quackle checkout lives (default: next to this folder).
set(QUACKLE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../quackle" CACHE PATH "Path to the quackle repo")

# Builds only the engine (libquackle). It does not need Qt.
add_subdirectory(${QUACKLE_DIR} ${CMAKE_BINARY_DIR}/libquackle EXCLUDE_FROM_ALL)

# pybind11: use the pip-installed copy if present, else a local clone.
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(COMMAND ${Python_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
                OUTPUT_VARIABLE PYBIND11_CMAKE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
if(PYBIND11_CMAKE_DIR)
  find_package(pybind11 CONFIG REQUIRED PATHS ${PYBIND11_CMAKE_DIR})
else()
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../pybind11 ${CMAKE_BINARY_DIR}/pybind11)
endif()

pybind11_add_module(qenv cpp/qenv.cpp ${QUACKLE_DIR}/test/trademarkedboards.cpp)
target_include_directories(qenv PRIVATE ${QUACKLE_DIR})
target_link_libraries(qenv PRIVATE libquackle)

# Put qenv*.so in the project folder so `import qenv` works from here.
set_target_properties(qenv PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Qt-free tool that builds a .gaddag from a .dawg (faster move generation).
add_executable(make_gaddag tools/make_gaddag.cpp)
set_target_properties(make_gaddag PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

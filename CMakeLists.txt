cmake_minimum_required(VERSION 3.20)

project(
  nanosentencepiece
  VERSION 0.1.0
  DESCRIPTION "A compact, SentencePiece-inspired BPE tokenizer in modern C++20"
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(NSP_BUILD_TESTS "Build nanosentencepiece tests" ON)
option(NSP_WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)
option(NSP_USE_GTEST "Use GoogleTest for nanosentencepiece tests" ON)

add_library(nanosentencepiece
  src/utf8.cpp
  src/normalization.cpp
  src/vocabulary.cpp
  src/model.cpp
  src/processor.cpp
  src/bpe_trainer.cpp
  src/tokenizer.cpp
  src/cli.cpp
)

target_include_directories(nanosentencepiece
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

if (MSVC)
  target_compile_options(nanosentencepiece PRIVATE /W4)
  if (NSP_WARNINGS_AS_ERRORS)
    target_compile_options(nanosentencepiece PRIVATE /WX)
  endif()
else()
  target_compile_options(nanosentencepiece PRIVATE -Wall -Wextra -Wpedantic)
  if (NSP_WARNINGS_AS_ERRORS)
    target_compile_options(nanosentencepiece PRIVATE -Werror)
  endif()
endif()

target_compile_features(nanosentencepiece PUBLIC cxx_std_20)

add_executable(nsp_train apps/nsp_train.cpp)
target_link_libraries(nsp_train PRIVATE nanosentencepiece)

add_executable(nsp_encode apps/nsp_encode.cpp)
target_link_libraries(nsp_encode PRIVATE nanosentencepiece)

add_executable(nsp_decode apps/nsp_decode.cpp)
target_link_libraries(nsp_decode PRIVATE nanosentencepiece)

add_executable(nsp_inspect apps/nsp_inspect.cpp)
target_link_libraries(nsp_inspect PRIVATE nanosentencepiece)

if (NSP_BUILD_TESTS)
  enable_testing()

  add_executable(nanosentencepiece_tests tests/test_main.cpp)
  target_link_libraries(nanosentencepiece_tests PRIVATE nanosentencepiece)

  if (NSP_USE_GTEST)
    find_package(GTest REQUIRED)
    include(GoogleTest)
    target_link_libraries(nanosentencepiece_tests PRIVATE GTest::gtest)
    set_property(
      TARGET nanosentencepiece_tests
      APPEND
      PROPERTY BUILD_RPATH
      "$<TARGET_FILE_DIR:GTest::gtest>"
    )
    gtest_discover_tests(nanosentencepiece_tests)
  else()
    add_test(NAME nanosentencepiece_tests COMMAND nanosentencepiece_tests)
  endif()
endif()

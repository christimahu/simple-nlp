cmake_minimum_required(VERSION 3.10)
project(NLP_SentimentAnalysis VERSION 1.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Define source directories
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)
set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
set(TEST_DIR ${CMAKE_SOURCE_DIR}/tests)

# Enable verbose output
set(CMAKE_VERBOSE_MAKEFILE ON)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Add include directories
include_directories(${INCLUDE_DIR})

# Create a separate object for ascii_word_cloud.cpp to debug issues
add_library(ascii_word_cloud_obj OBJECT ${SRC_DIR}/ascii_word_cloud.cpp)
target_include_directories(ascii_word_cloud_obj PRIVATE ${INCLUDE_DIR})

# Library source files
set(LIB_SOURCES
    ${SRC_DIR}/text_preprocessor.cpp
    ${SRC_DIR}/tfidf_vectorizer.cpp
    ${SRC_DIR}/sgd_classifier.cpp
    ${SRC_DIR}/model_evaluator.cpp
    ${SRC_DIR}/sentiment_analyzer.cpp
)

# Create shared library with explicit inclusion of ascii_word_cloud.cpp
add_library(nlp_sentiment SHARED ${LIB_SOURCES} $<TARGET_OBJECTS:ascii_word_cloud_obj>)

# Set library properties
set_target_properties(nlp_sentiment PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    POSITION_INDEPENDENT_CODE ON
)

# Main application
add_executable(nlp_main ${SRC_DIR}/nlp_main.cpp)
target_link_libraries(nlp_main PRIVATE nlp_sentiment)

# Add test executable
add_executable(nlp_tests ${TEST_DIR}/nlp_tests.cpp)
target_link_libraries(nlp_tests PRIVATE nlp_sentiment)

# Enable testing
enable_testing()
add_test(NAME unit_tests COMMAND nlp_tests)

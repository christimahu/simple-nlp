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

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Add include directories
include_directories(${INCLUDE_DIR})

# Library source files
set(LIB_SOURCES
    ${SRC_DIR}/text_preprocessor.cpp
    ${SRC_DIR}/tfidf_vectorizer.cpp
    ${SRC_DIR}/sgd_classifier.cpp
    ${SRC_DIR}/model_evaluator.cpp
    ${SRC_DIR}/ascii_word_cloud.cpp
    ${SRC_DIR}/sentiment_analyzer.cpp
)

# Create shared library
add_library(nlp_sentiment SHARED ${LIB_SOURCES})

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

# Installation rules
install(TARGETS nlp_sentiment
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(TARGETS nlp_main RUNTIME DESTINATION bin)
install(FILES ${INCLUDE_DIR}/sentiment_analysis.h DESTINATION include)

# Package generation with CPack
include(CPack)

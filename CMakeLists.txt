# @file CMakeLists.txt
# @brief Main CMake configuration for the NLP Sentiment Analysis project
#
# This file configures the build system for the C++ sentiment analysis library.
# It defines targets, dependencies, and installation rules for the entire project.
# The project structure follows modern C++ best practices with clear separation
# between interface and implementation.

cmake_minimum_required(VERSION 3.14)
project(NLP_SentimentAnalysis VERSION 1.0)

# Set C++23 standard for modern features like concepts, ranges, and std::span
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Option to build as a shared library (default) or static library
option(BUILD_SHARED_LIBS "Build as shared library" ON)

# Define source directories for better organization
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)
set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
set(TEST_DIR ${CMAKE_SOURCE_DIR}/tests)
set(EXAMPLES_DIR ${CMAKE_SOURCE_DIR}/examples)

# Enable verbose output for debugging the build process
set(CMAKE_VERBOSE_MAKEFILE ON)

# Set output directories for binaries and libraries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Add include directories for header files
include_directories(${INCLUDE_DIR})

# Define the source files for the library
set(LIB_SOURCES
    ${SRC_DIR}/text_preprocessor.cpp
    ${SRC_DIR}/tfidf_vectorizer.cpp
    ${SRC_DIR}/sgd_classifier.cpp
    ${SRC_DIR}/model_evaluator.cpp
    ${SRC_DIR}/ascii_word_cloud.cpp
    ${SRC_DIR}/sentiment_analyzer.cpp
    ${SRC_DIR}/sentiment_dataset.cpp
)

# Create the main library target
add_library(nlp_sentiment ${LIB_SOURCES})

# Set library properties including versioning and position-independent code
set_target_properties(nlp_sentiment PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    POSITION_INDEPENDENT_CODE ON
    EXPORT_NAME NLPSentiment
)

# Install the library for system-wide use
install(TARGETS nlp_sentiment
    EXPORT NLPSentimentTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

# Install header files for developers to use the library
install(FILES 
    ${INCLUDE_DIR}/sentiment_analysis.h
    ${INCLUDE_DIR}/text_preprocessor.h
    ${INCLUDE_DIR}/tfidf_vectorizer.h
    ${INCLUDE_DIR}/sgd_classifier.h
    ${INCLUDE_DIR}/model_evaluator.h
    ${INCLUDE_DIR}/ascii_word_cloud.h
    ${INCLUDE_DIR}/classifier_model.h
    ${INCLUDE_DIR}/sentiment_dataset.h
    DESTINATION include
)

# Create export package for easy integration in other CMake projects
install(EXPORT NLPSentimentTargets
    FILE NLPSentimentTargets.cmake
    NAMESPACE NLP::
    DESTINATION lib/cmake/NLPSentiment
)

# Add the tests
add_executable(text_preprocessor_test ${TEST_DIR}/text_preprocessor_test.cpp)
target_link_libraries(text_preprocessor_test PRIVATE nlp_sentiment)

add_executable(tfidf_vectorizer_test ${TEST_DIR}/tfidf_vectorizer_test.cpp)
target_link_libraries(tfidf_vectorizer_test PRIVATE nlp_sentiment)

add_executable(sgd_classifier_test ${TEST_DIR}/sgd_classifier_test.cpp)
target_link_libraries(sgd_classifier_test PRIVATE nlp_sentiment)

add_executable(model_evaluator_test ${TEST_DIR}/model_evaluator_test.cpp)
target_link_libraries(model_evaluator_test PRIVATE nlp_sentiment)

add_executable(sentiment_analyzer_test ${TEST_DIR}/sentiment_analyzer_test.cpp)
target_link_libraries(sentiment_analyzer_test PRIVATE nlp_sentiment)

# Enable testing
enable_testing()
add_test(NAME text_preprocessor_test COMMAND text_preprocessor_test)
add_test(NAME tfidf_vectorizer_test COMMAND tfidf_vectorizer_test)
add_test(NAME sgd_classifier_test COMMAND sgd_classifier_test)
add_test(NAME model_evaluator_test COMMAND model_evaluator_test)
add_test(NAME sentiment_analyzer_test COMMAND sentiment_analyzer_test)

# Add the example applications
add_executable(simple_sentiment ${EXAMPLES_DIR}/simple_sentiment.cpp)
target_link_libraries(simple_sentiment PRIVATE nlp_sentiment)

add_executable(word_cloud ${EXAMPLES_DIR}/word_cloud.cpp)
target_link_libraries(word_cloud PRIVATE nlp_sentiment)

add_executable(file_analyzer ${EXAMPLES_DIR}/file_analyzer.cpp)
target_link_libraries(file_analyzer PRIVATE nlp_sentiment)

add_executable(run_pipeline ${EXAMPLES_DIR}/run_pipeline.cpp)
target_link_libraries(run_pipeline PRIVATE nlp_sentiment)

# Generate and install CMake package config
include(CMakePackageConfigHelpers)

# Write version file for dependency management
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/NLPSentimentConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

# Configure package config file
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/NLPSentimentConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/NLPSentimentConfig.cmake"
    INSTALL_DESTINATION lib/cmake/NLPSentiment
)

# Install config files
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/NLPSentimentConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/NLPSentimentConfigVersion.cmake"
    DESTINATION lib/cmake/NLPSentiment
)

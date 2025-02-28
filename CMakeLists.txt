# @file CMakeLists.txt
# @brief Main CMake configuration file for the NLP Sentiment Analysis library
# 
# This file configures the build system for the sentiment analysis library,
# defining targets, dependencies, and installation rules. It demonstrates
# modern CMake practices and a clean organization of build components.

cmake_minimum_required(VERSION 3.14)
project(NLP_SentimentAnalysis VERSION 1.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Option to build as a shared library
option(BUILD_SHARED_LIBS "Build as shared library" ON)

# Define source directories
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)
set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
set(TEST_DIR ${CMAKE_SOURCE_DIR}/tests)
set(EXAMPLES_DIR ${CMAKE_SOURCE_DIR}/examples)

# Enable verbose output for debugging
set(CMAKE_VERBOSE_MAKEFILE ON)

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
    ${SRC_DIR}/sentiment_dataset.cpp
)

# Create the main library target
add_library(nlp_sentiment ${LIB_SOURCES})

# Set library properties
set_target_properties(nlp_sentiment PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    POSITION_INDEPENDENT_CODE ON
    EXPORT_NAME NLPSentiment
)

# Add library installation targets
install(TARGETS nlp_sentiment
    EXPORT NLPSentimentTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

# Install header files
install(FILES ${INCLUDE_DIR}/sentiment_analysis.h
    DESTINATION include
)

# Create export package
install(EXPORT NLPSentimentTargets
    FILE NLPSentimentTargets.cmake
    NAMESPACE NLP::
    DESTINATION lib/cmake/NLPSentiment
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

# Configuration for more specific platform dependencies
if(WIN32)
    # Windows-specific configuration
    target_compile_definitions(nlp_sentiment PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
elseif(UNIX AND NOT APPLE)
    # Linux-specific configuration
    target_link_libraries(nlp_sentiment PRIVATE pthread)
endif()

# Package generation with CPack
include(CPack)

# Generate and install CMake package config
include(CMakePackageConfigHelpers)

# Write version file
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

# Add subdirectories for other components
add_subdirectory(examples)

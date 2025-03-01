# @file examples/CMakeLists.txt
# @brief CMake configuration for example applications
# 
# This file configures the build for example applications that
# demonstrate the usage of the sentiment analysis library.
# Each example showcases different features of the library
# with specific use cases and scenarios.

# Simple sentiment analysis example - Basic usage of the library
add_executable(example_simple_sentiment simple_sentiment.cpp)
target_link_libraries(example_simple_sentiment PRIVATE nlp_sentiment)

# Word cloud visualization example - Demonstrates text visualization capabilities
add_executable(example_word_cloud word_cloud.cpp)
target_link_libraries(example_word_cloud PRIVATE nlp_sentiment)

# Advanced example with file I/O - Shows how to analyze text from files
add_executable(example_file_analyzer file_analyzer.cpp)
target_link_libraries(example_file_analyzer PRIVATE nlp_sentiment)

# End-to-end pipeline example - Demonstrates the complete analysis workflow
add_executable(example_run_pipeline run_pipeline.cpp)
target_link_libraries(example_run_pipeline PRIVATE nlp_sentiment)

# Install examples to make them available system-wide
install(TARGETS 
    example_simple_sentiment
    example_word_cloud
    example_file_analyzer
    example_run_pipeline
    RUNTIME DESTINATION bin/examples
)

# Install example source code for reference and learning
install(FILES
    simple_sentiment.cpp
    word_cloud.cpp
    file_analyzer.cpp
    run_pipeline.cpp
    DESTINATION share/nlp_sentiment/examples
)

# Add custom target to run all examples sequentially for demonstration
add_custom_target(run_all_examples
    COMMAND example_simple_sentiment
    COMMAND example_word_cloud
    COMMAND example_file_analyzer
    COMMAND example_run_pipeline
    DEPENDS 
        example_simple_sentiment
        example_word_cloud 
        example_file_analyzer
        example_run_pipeline
    COMMENT "Running all sentiment analysis examples"
    VERBATIM
)

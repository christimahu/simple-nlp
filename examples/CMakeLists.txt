# @file examples/CMakeLists.txt
# @brief CMake configuration for example applications
# 
# This file configures the build for example applications that
# demonstrate the usage of the sentiment analysis library.

# Simple sentiment analysis example
add_executable(example_simple_sentiment simple_sentiment.cpp)
target_link_libraries(example_simple_sentiment PRIVATE nlp_sentiment)

# Word cloud visualization example
add_executable(example_word_cloud word_cloud.cpp)
target_link_libraries(example_word_cloud PRIVATE nlp_sentiment)

# Advanced example with file I/O
add_executable(example_file_analyzer file_analyzer.cpp)
target_link_libraries(example_file_analyzer PRIVATE nlp_sentiment)

# Install examples
install(TARGETS 
    example_simple_sentiment
    example_word_cloud
    example_file_analyzer
    RUNTIME DESTINATION bin/examples
)

# Install example source code for reference
install(FILES
    simple_sentiment.cpp
    word_cloud.cpp
    file_analyzer.cpp
    DESTINATION share/nlp_sentiment/examples
)

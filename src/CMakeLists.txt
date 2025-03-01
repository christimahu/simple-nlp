# @file src/CMakeLists.txt
# @brief CMake configuration for the library source files
#
# This file configures the build specifically for the library's
# implementation files. It defines compilation options and
# dependencies for the source code components.

# Enable additional warnings to ensure code quality
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -Werror=return-type)
elseif(MSVC)
    add_compile_options(/W4 /WX)
endif()

# Add include directories for implementation files
include_directories(
    ${CMAKE_SOURCE_DIR}/include
)

# List all source files in this directory
set(SOURCE_FILES
    text_preprocessor.cpp
    tfidf_vectorizer.cpp
    sgd_classifier.cpp
    model_evaluator.cpp
    ascii_word_cloud.cpp
    sentiment_analyzer.cpp
    sentiment_dataset.cpp
)

# Define a library target with all source files
# This is not used directly but helps with organization
add_library(nlp_sentiment_src OBJECT ${SOURCE_FILES})

# Set properties for the source component
set_target_properties(nlp_sentiment_src PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# Add custom analysis targets for code quality
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # Add clang-tidy target if available
    find_program(CLANG_TIDY clang-tidy)
    if(CLANG_TIDY)
        add_custom_target(tidy
            COMMAND ${CLANG_TIDY} ${SOURCE_FILES} -- -std=c++23 -I${CMAKE_SOURCE_DIR}/include
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Running clang-tidy on source files"
            VERBATIM
        )
    endif()
endif()

# Add custom target for documentation generation if Doxygen is available
find_package(Doxygen)
if(DOXYGEN_FOUND)
    add_custom_target(source_docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_SOURCE_DIR}/doc/Doxyfile
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Generating source code documentation with Doxygen"
        VERBATIM
    )
endif()

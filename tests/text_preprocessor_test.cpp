/**
 * @file text_preprocessor_test.cpp
 * @brief Tests for the TextPreprocessor component
 */

#include "text_preprocessor.h"
#include <iostream>
#include <cassert>
#include <string>

// Simple test framework
#define TEST(name) void name()
#define RUN_TEST(name) std::cout << "Running " << #name << "... "; name(); std::cout << "PASSED" << std::endl

/**
 * Test TextPreprocessor functionality.
 * 
 * This function tests the text preprocessing component, ensuring
 * that it properly cleans and normalizes text data according to
 * expectations. It uses a functional approach with recursive checks.
 */
TEST(testTextPreprocessor) {
    nlp::TextPreprocessor preprocessor;
    
    // Test with a typical tweet
    std::string input = "@user I LOVED the movie!!! It's amazing & worth $12.99 :) #mustwatch";
    std::string processed = preprocessor.preprocess(input);
    
    // Check basic expectations
    assert(!processed.empty());
    
    // Recursively check if all characters are lowercase
    const auto checkAllLowercase = [](auto& self, const std::string& text, size_t index) -> bool {
        // Base case: all characters checked
        if (index >= text.size()) {
            return true;
        }
        
        // Check if this character is uppercase
        if (std::isupper(static_cast<unsigned char>(text[index]))) {
            return false;
        }
        
        // Recursively check next character
        return self(self, text, index + 1);
    };
    
    assert(checkAllLowercase(checkAllLowercase, processed, 0) && "Text should be lowercase");
    
    // Recursively check for specific characters
    const auto checkForChar = [](auto& self, const std::string& text, char target, size_t index) -> bool {
        // Base case: character not found in remaining text
        if (index >= text.size()) {
            return false;
        }
        
        // Check if this is the target character
        if (text[index] == target) {
            return true;
        }
        
        // Recursively check next character
        return self(self, text, target, index + 1);
    };
    
    // Should not contain punctuation
    assert(!checkForChar(checkForChar, processed, '!', 0));
    assert(!checkForChar(checkForChar, processed, '@', 0));
    assert(!checkForChar(checkForChar, processed, '#', 0));
    assert(!checkForChar(checkForChar, processed, '$', 0));
    
    // Should not contain numbers
    assert(!checkForChar(checkForChar, processed, '1', 0));
    assert(!checkForChar(checkForChar, processed, '2', 0));
    assert(!checkForChar(checkForChar, processed, '9', 0));
    
    // Test with empty string
    assert(preprocessor.preprocess("").empty());
    
    // Test with only stopwords
    std::string allStopwords = "the and a of";
    std::string processedStopwords = preprocessor.preprocess(allStopwords);
    assert(processedStopwords.empty());
    
    // Test individual preprocessing functions
    auto preprocessingFuncs = preprocessor.getPreprocessingFunctions();
    
    // Test lowercase function
    std::string upperText = "ALL UPPERCASE TEXT";
    std::string lowerText = preprocessingFuncs["lowercase"](upperText);
    assert(checkAllLowercase(checkAllLowercase, lowerText, 0) && "Lowercase function failed");
    
    // Test remove_punctuation function
    std::string punctText = "Text, with. punctuation!";
    std::string noPunctText = preprocessingFuncs["remove_punctuation"](punctText);
    assert(!checkForChar(checkForChar, noPunctText, ',', 0) && "Punctuation removal failed");
    assert(!checkForChar(checkForChar, noPunctText, '.', 0) && "Punctuation removal failed");
    assert(!checkForChar(checkForChar, noPunctText, '!', 0) && "Punctuation removal failed");
}

int main() {
    std::cout << "==== TextPreprocessor Tests ====" << std::endl;
    testTextPreprocessor();
    std::cout << "All TextPreprocessor tests passed!" << std::endl;
    return 0;
}

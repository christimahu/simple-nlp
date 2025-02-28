/**
 * @file text_preprocessor.cpp
 * @brief Implementation of the TextPreprocessor class
 * 
 * This file contains the implementation of text preprocessing methods,
 * which clean and normalize text for sentiment analysis.
 */

#include "sentiment_analysis.h"
#include <algorithm>
#include <cctype>
#include <sstream>
#include <locale>
#include <functional>

namespace nlp {

/**
 * Constructor initializes the set of stopwords.
 * 
 * Stopwords are common words (like "the", "a", "is") that are often
 * removed from text during preprocessing as they carry little semantic meaning.
 */
TextPreprocessor::TextPreprocessor() {
    // Initialize a comprehensive list of English stopwords
    stops = {
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", 
        "you're", "you've", "you'll", "you'd", "your", "yours", "yourself", 
        "yourselves", "he", "him", "his", "himself", "she", "she's", "her", 
        "hers", "herself", "it", "it's", "its", "itself", "they", "them", "their", 
        "theirs", "themselves", "what", "which", "who", "whom", "this", "that", 
        "that'll", "these", "those", "am", "is", "are", "was", "were", "be", 
        "been", "being", "have", "has", "had", "having", "do", "does", "did", 
        "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", 
        "until", "while", "of", "at", "by", "for", "with", "about", "against", 
        "between", "into", "through", "during", "before", "after", "above", 
        "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", 
        "under", "again", "further", "then", "once", "here", "there", "when", 
        "where", "why", "how", "all", "any", "both", "each", "few", "more", 
        "most", "other", "some", "such", "no", "nor", "not", "only", "own", 
        "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", 
        "don", "don't", "should", "should've", "now", "d", "ll", "m", "o", 
        "re", "ve", "y"
    };
}

/**
 * Main preprocessing function that applies a series of text transformations.
 * 
 * This function uses a functional approach to apply a sequence of transformations
 * to the input text. If specific steps are provided, only those are applied;
 * otherwise, a default sequence is used.
 * 
 * @param text The input text to preprocess
 * @param steps List of preprocessing steps to apply (empty means use defaults)
 * @return The preprocessed text
 */
std::string TextPreprocessor::preprocess(std::string_view text, 
                                        const std::vector<std::string>& steps) const {
    // Get all available preprocessing functions
    auto preprocessingFuncs = getPreprocessingFunctions();
    
    // Start with a copy of the text as string
    std::string result{text};
    
    // If no steps provided, apply default steps in a sensible order
    if (steps.empty()) {
        // Define sensible default order for preprocessing
        static const std::vector<std::string> defaultSteps = {
            "remove_non_ascii", 
            "lowercase", 
            "remove_punctuation", 
            "remove_numbers", 
            "strip_whitespace", 
            "remove_stopwords"
        };
        
        // Apply each step functionally, piping the output of one to the input of the next
        for (const auto& step : defaultSteps) {
            if (preprocessingFuncs.contains(step)) {
                result = preprocessingFuncs.at(step)(result);
            }
        }
    } else {
        // Apply only the specified steps in the given order
        for (const auto& step : steps) {
            if (preprocessingFuncs.contains(step)) {
                result = preprocessingFuncs.at(step)(result);
            }
        }
    }
    
    return result;
}

/**
 * Returns a map of preprocessing function names to function objects.
 * 
 * This function demonstrates function composition and the use of lambda functions
 * to create a mapping of named preprocessing steps that can be dynamically selected.
 * 
 * @return Map of preprocessing function names to function objects
 */
std::unordered_map<std::string, TextPreprocessor::PreprocessingFunc> 
TextPreprocessor::getPreprocessingFunctions() const {
    // Create and return a map linking step names to their implementations
    return {
        {"remove_non_ascii", [this](std::string_view text) { return removeNonAscii(text); }},
        {"lowercase", [this](std::string_view text) { return lowercase(text); }},
        {"remove_punctuation", [this](std::string_view text) { return removePunctuation(text); }},
        {"remove_numbers", [this](std::string_view text) { return removeNumbers(text); }},
        {"strip_whitespace", [this](std::string_view text) { return stripWhitespace(text); }},
        {"remove_stopwords", [this](std::string_view text) { return removeStopwords(text); }},
        {"stem_words", [this](std::string_view text) { return stemWords(text); }}
    };
}

/**
 * Removes non-ASCII characters from text.
 * 
 * Uses a functional approach with ranges to filter characters.
 * 
 * @param text Input text
 * @return Text with only ASCII characters
 */
std::string TextPreprocessor::removeNonAscii(std::string_view text) const {
    std::string result;
    result.reserve(text.size());
    
    // Use ranges to filter ASCII characters (functional approach)
    auto asciiOnly = text | std::views::filter([](char c) { 
        return static_cast<unsigned char>(c) < 128; 
    });
    
    // Copy filtered characters to result
    std::ranges::copy(asciiOnly, std::back_inserter(result));
    
    return result;
}

/**
 * Converts text to lowercase.
 * 
 * Uses a functional approach with ranges to transform characters.
 * 
 * @param text Input text
 * @return Lowercase text
 */
std::string TextPreprocessor::lowercase(std::string_view text) const {
    std::string result{text};
    
    // Use ranges to transform all characters to lowercase (functional approach)
    std::ranges::transform(result, result.begin(), 
                         [](unsigned char c) { return std::tolower(c); });
    
    return result;
}

/**
 * Removes punctuation characters from text.
 * 
 * Uses a functional approach with ranges to filter characters.
 * 
 * @param text Input text
 * @return Text without punctuation
 */
std::string TextPreprocessor::removePunctuation(std::string_view text) const {
    std::string result;
    result.reserve(text.size());
    
    // Use ranges to filter non-punctuation characters (functional approach)
    auto noPunct = text | std::views::filter([](char c) { 
        return !std::ispunct(static_cast<unsigned char>(c)); 
    });
    
    // Copy filtered characters to result
    std::ranges::copy(noPunct, std::back_inserter(result));
    
    return result;
}

/**
 * Removes numeric digits from text.
 * 
 * Uses a functional approach with ranges to filter characters.
 * 
 * @param text Input text
 * @return Text without digits
 */
std::string TextPreprocessor::removeNumbers(std::string_view text) const {
    std::string result;
    result.reserve(text.size());
    
    // Use ranges to filter non-digit characters (functional approach)
    auto noDigits = text | std::views::filter([](char c) { 
        return !std::isdigit(static_cast<unsigned char>(c)); 
    });
    
    // Copy filtered characters to result
    std::ranges::copy(noDigits, std::back_inserter(result));
    
    return result;
}

/**
 * Normalizes whitespace in text, removing leading/trailing and duplicate spaces.
 * 
 * @param text Input text
 * @return Text with normalized whitespace
 */
std::string TextPreprocessor::stripWhitespace(std::string_view text) const {
    // Convert to string for manipulation
    std::string str{text};
    
    // Skip if empty
    if (str.empty()) {
        return str;
    }
    
    // Remove leading whitespace
    auto start = str.find_first_not_of(" \t\n\r\f\v");
    if (start == std::string::npos) {
        return "";
    }
    
    // Remove trailing whitespace
    auto end = str.find_last_not_of(" \t\n\r\f\v");
    str = str.substr(start, end - start + 1);
    
    // Replace multiple whitespace with a single space using a recursive approach
    const auto stripExtraWhitespace = [](const std::string& s, std::string result = "", bool prevWasSpace = false) -> std::string {
        // Base case: end of string
        if (s.empty()) {
            return result;
        }
        
        // Process first character and recursively process the rest
        char c = s[0];
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (!prevWasSpace) {
                return stripExtraWhitespace(s.substr(1), result + ' ', true);
            } else {
                return stripExtraWhitespace(s.substr(1), result, true);
            }
        } else {
            return stripExtraWhitespace(s.substr(1), result + c, false);
        }
    };
    
    return stripExtraWhitespace(str);
}

/**
 * Removes common stopwords from text.
 * 
 * Uses a functional approach with tokenization and filtering.
 * 
 * @param text Input text
 * @return Text without stopwords
 */
std::string TextPreprocessor::removeStopwords(std::string_view text) const {
    std::stringstream ss{std::string{text}};
    std::vector<std::string> words;
    std::string word;
    
    // Tokenize the text into words
    while (ss >> word) {
        words.push_back(word);
    }
    
    // Filter out stopwords using a functional approach
    auto nonStopwords = words | std::views::filter([this](const std::string& w) {
        return stops.find(w) == stops.end();
    });
    
    // Use a recursive approach to join the filtered words
    const auto joinWords = [](auto begin, auto end, std::string result = "") -> std::string {
        // Base case: no more words
        if (begin == end) {
            return result;
        }
        
        // Add a space if this isn't the first word
        if (!result.empty()) {
            result += " ";
        }
        
        // Add this word and recursively process the rest
        return joinWords(std::next(begin), end, result + *begin);
    };
    
    // Create a vector from the filtered view for easier manipulation
    std::vector<std::string> filteredWords;
    std::ranges::copy(nonStopwords, std::back_inserter(filteredWords));
    
    return joinWords(filteredWords.begin(), filteredWords.end());
}

/**
 * Simple stemming function (placeholder for a more sophisticated implementation).
 * 
 * Note: A real implementation would typically use a Porter stemmer or similar.
 * 
 * @param text Input text
 * @return Stemmed text
 */
std::string TextPreprocessor::stemWords(std::string_view text) const {
    // This would require a Porter stemmer or similar library
    // For now, just return the original text
    return std::string{text};
}

} // namespace nlp

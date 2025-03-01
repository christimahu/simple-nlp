/**
 * @file text_preprocessor.cpp
 * @brief Implements text preprocessing functions for sentiment analysis.
 *
 * This file provides the implementation of text preprocessing utilities
 * that clean and normalize raw text data before feature extraction.
 * It applies transformations like lowercasing, removing punctuation,
 * and removing stopwords to standardize text for machine learning.
 */

#include "text_preprocessor.h"
#include <algorithm>
#include <cctype>
#include <sstream>
#include <string>
#include <unordered_set>

namespace nlp {

/**
 * @brief Constructs a new TextPreprocessor instance with default stopwords.
 */
TextPreprocessor::TextPreprocessor() {
    // Initialize common English stopwords
    stopwords = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
        "to", "was", "were", "will", "with", "i", "me", "my", "myself",
        "we", "our", "ours", "ourselves", "you", "your", "yours",
        "yourself", "yourselves", "he", "him", "his", "himself", "she",
        "her", "hers", "herself", "it", "its", "itself", "they", "them",
        "their", "theirs", "themselves", "what", "which", "who", "whom",
        "this", "that", "these", "those", "am", "is", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "having", "do", "does",
        "did", "doing", "would", "should", "could", "ought", "i'm", "you're",
        "he's", "she's", "it's", "we're", "they're", "i've", "you've",
        "we've", "they've", "i'd", "you'd", "he'd", "she'd", "we'd",
        "they'd", "i'll", "you'll", "he'll", "she'll", "we'll", "they'll",
        "isn't", "aren't", "wasn't", "weren't", "hasn't", "haven't", "hadn't",
        "doesn't", "don't", "didn't", "won't", "wouldn't", "shan't", "shouldn't",
        "can't", "cannot", "couldn't", "mustn't", "let's", "that's", "who's",
        "what's", "here's", "there's", "when's", "where's", "why's", "how's",
        "so", "than", "too", "very", "just", "but", "however", "still"
    };
}

/**
 * @brief Applies a sequence of preprocessing steps to clean and normalize text.
 * 
 * This method applies text transformations to prepare the input for machine learning.
 * If no specific steps are provided, it applies a default sequence of transformations.
 * 
 * @param text The input text to process.
 * @param steps The set of transformations to apply.
 * @return The preprocessed text.
 */
std::string TextPreprocessor::preprocess(std::string_view text, const std::vector<std::string>& steps) const {
    // If no steps provided, use default preprocessing pipeline
    std::vector<std::string> preprocessingSteps = steps;
    if (preprocessingSteps.empty()) {
        preprocessingSteps = {
            "remove_non_ascii", 
            "lowercase", 
            "remove_punctuation", 
            "remove_numbers", 
            "strip_whitespace", 
            "remove_stopwords"
        };
    }
    
    std::string processedText(text);
    auto functions = getPreprocessingFunctions();
    
    // Apply each preprocessing step in sequence
    for (const auto& step : preprocessingSteps) {
        auto it = functions.find(step);
        if (it != functions.end()) {
            processedText = it->second(processedText);
        }
    }
    
    return processedText;
}

/**
 * @brief Creates a map of preprocessing function names to their implementations.
 * 
 * This method returns all available preprocessing functions, allowing
 * for dynamic construction of preprocessing pipelines and easy testing.
 * 
 * @return Map of function names to function implementations.
 */
std::unordered_map<std::string, TextPreprocessor::PreprocessingFunc> 
TextPreprocessor::getPreprocessingFunctions() const {
    return {
        {"lowercase", [this](std::string_view t) { return this->lowercase(t); }},
        {"remove_punctuation", [this](std::string_view t) { return this->removePunctuation(t); }},
        {"remove_numbers", [this](std::string_view t) { return this->removeNumbers(t); }},
        {"remove_non_ascii", [this](std::string_view t) { return this->removeNonAscii(t); }},
        {"strip_whitespace", [this](std::string_view t) { return this->stripWhitespace(t); }},
        {"remove_stopwords", [this](std::string_view t) { return this->removeStopwords(t); }},
        {"stem_words", [this](std::string_view t) { return this->stemWords(t); }}
    };
}

/**
 * @brief Converts text to lowercase.
 * 
 * This transformation ensures all characters are lowercase,
 * which helps with standardization and reduces feature dimensionality.
 * 
 * @param text The input text.
 * @return The lowercase version of the text.
 */
std::string TextPreprocessor::lowercase(std::string_view text) const {
    std::string result(text);
    std::transform(result.begin(), result.end(), result.begin(), 
                   [](unsigned char c) { return std::tolower(c); });
    return result;
}

/**
 * @brief Removes punctuation from text.
 * 
 * This transformation strips all punctuation marks from the text,
 * which helps with tokenization and reduces noise in the feature set.
 * 
 * @param text The input text.
 * @return Text without punctuation.
 */
std::string TextPreprocessor::removePunctuation(std::string_view text) const {
    std::string result;
    result.reserve(text.size());
    
    for (char c : text) {
        if (!std::ispunct(static_cast<unsigned char>(c))) {
            result += c;
        }
    }
    return result;
}

/**
 * @brief Removes numerical digits from text.
 * 
 * This transformation removes all numeric characters (0-9) from the input,
 * as numbers typically don't contribute to sentiment analysis.
 * 
 * @param text The input text.
 * @return Text without numbers.
 */
std::string TextPreprocessor::removeNumbers(std::string_view text) const {
    std::string result;
    result.reserve(text.size());
    
    for (char c : text) {
        if (!std::isdigit(static_cast<unsigned char>(c))) {
            result += c;
        }
    }
    return result;
}

/**
 * @brief Removes non-ASCII characters from text.
 * 
 * This transformation ensures that only standard ASCII characters are retained,
 * which can help with compatibility and reduce noise from special characters.
 * 
 * @param text The input text.
 * @return Cleaned text with only ASCII characters.
 */
std::string TextPreprocessor::removeNonAscii(std::string_view text) const {
    std::string result;
    result.reserve(text.size());
    
    for (char c : text) {
        if (static_cast<unsigned char>(c) < 128) {
            result += c;
        }
    }
    return result;
}

/**
 * @brief Normalizes whitespace in text.
 * 
 * This transformation collapses multiple consecutive whitespace characters
 * into a single space and trims leading/trailing whitespace.
 * 
 * @param text The input text.
 * @return Trimmed text with normalized whitespace.
 */
std::string TextPreprocessor::stripWhitespace(std::string_view text) const {
    std::string result;
    result.reserve(text.size());
    
    bool lastWasSpace = true;  // Start with true to trim leading spaces
    
    for (char c : text) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (!lastWasSpace) {
                result += ' ';
            }
            lastWasSpace = true;
        } else {
            result += c;
            lastWasSpace = false;
        }
    }
    
    // Trim trailing space if present
    if (!result.empty() && result.back() == ' ') {
        result.pop_back();
    }
    
    return result;
}

/**
 * @brief Removes common stopwords from text.
 * 
 * This transformation removes common words (like "the", "is", "and")
 * that typically don't contribute meaningful sentiment information.
 * 
 * @param text The input text.
 * @return The text with stopwords removed.
 */
std::string TextPreprocessor::removeStopwords(std::string_view text) const {
    std::istringstream iss{std::string(text)};
    std::ostringstream oss;
    std::string word;
    bool firstWord = true;
    
    while (iss >> word) {
        if (stopwords.find(word) == stopwords.end()) {
            if (!firstWord) {
                oss << ' ';
            }
            oss << word;
            firstWord = false;
        }
    }
    
    return oss.str();
}

/**
 * @brief Applies basic stemming to reduce words to their root form.
 * 
 * This is a simple stemming implementation that handles common English
 * suffixes. For production use, consider a more sophisticated stemming
 * algorithm like Porter or Snowball.
 * 
 * @param text The input text.
 * @return The stemmed version of the text.
 */
std::string TextPreprocessor::stemWords(std::string_view text) const {
    // This is a placeholder for a more sophisticated stemming algorithm
    // A simple implementation might handle common suffixes
    
    std::istringstream iss{std::string(text)};
    std::ostringstream oss;
    std::string word;
    bool firstWord = true;
    
    while (iss >> word) {
        // Very basic stemming for common English suffixes
        size_t len = word.length();
        
        if (len > 3) {
            // Handle -ing suffix
            if (len > 4 && word.substr(len-3) == "ing") {
                if (len > 5 && std::isalpha(word[len-4]) && std::isalpha(word[len-5])) {
                    word = word.substr(0, len-3);
                }
            }
            // Handle -ed suffix
            else if (len > 3 && word.substr(len-2) == "ed") {
                if (len > 4 && std::isalpha(word[len-3]) && std::isalpha(word[len-4])) {
                    word = word.substr(0, len-2);
                }
            }
            // Handle -s suffix (plural)
            else if (word.back() == 's' && len > 3 && std::isalpha(word[len-2])) {
                word = word.substr(0, len-1);
            }
        }
        
        if (!firstWord) {
            oss << ' ';
        }
        oss << word;
        firstWord = false;
    }
    
    return oss.str();
}

}  // namespace nlp

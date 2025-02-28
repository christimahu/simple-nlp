#include "sentiment_analysis.h"
#include <algorithm>
#include <cctype>
#include <sstream>
#include <locale>

namespace nlp {

TextPreprocessor::TextPreprocessor() {
    // Initialize basic English stopwords
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

std::string TextPreprocessor::preprocess(const std::string& text, const std::vector<std::string>& steps) {
    std::string result = text;
    
    // If no steps provided, apply all steps
    if (steps.empty()) {
        result = removeNonAscii(result);
        result = lowercase(result);
        result = removePunctuation(result);
        result = removeNumbers(result);
        result = stripWhitespace(result);
        result = removeStopwords(result);
        // Note: stemWords would be here if we had a stemming library
    } else {
        // Apply only the specified steps
        for (const auto& step : steps) {
            if (step == "remove_non_ascii") {
                result = removeNonAscii(result);
            } else if (step == "lowercase") {
                result = lowercase(result);
            } else if (step == "remove_punctuation") {
                result = removePunctuation(result);
            } else if (step == "remove_numbers") {
                result = removeNumbers(result);
            } else if (step == "strip_whitespace") {
                result = stripWhitespace(result);
            } else if (step == "remove_stopwords") {
                result = removeStopwords(result);
            } else if (step == "stem_words") {
                result = stemWords(result);
            }
        }
    }
    
    return result;
}

std::string TextPreprocessor::removeNonAscii(const std::string& text) {
    std::string result;
    for (char c : text) {
        if (static_cast<unsigned char>(c) < 128) {
            result += c;
        }
    }
    return result;
}

std::string TextPreprocessor::lowercase(const std::string& text) {
    std::string result = text;
    std::transform(result.begin(), result.end(), result.begin(),
                  [](unsigned char c){ return std::tolower(c); });
    return result;
}

std::string TextPreprocessor::removePunctuation(const std::string& text) {
    std::string result;
    for (char c : text) {
        if (!std::ispunct(static_cast<unsigned char>(c))) {
            result += c;
        }
    }
    return result;
}

std::string TextPreprocessor::removeNumbers(const std::string& text) {
    std::string result;
    for (char c : text) {
        if (!std::isdigit(static_cast<unsigned char>(c))) {
            result += c;
        }
    }
    return result;
}

std::string TextPreprocessor::stripWhitespace(const std::string& text) {
    std::string result = text;
    
    // Remove leading and trailing whitespace
    auto start = result.find_first_not_of(" \t\n\r\f\v");
    if (start == std::string::npos) {
        return "";
    }
    
    auto end = result.find_last_not_of(" \t\n\r\f\v");
    result = result.substr(start, end - start + 1);
    
    // Replace multiple whitespace with a single space
    std::string finalResult;
    bool prevWasSpace = false;
    
    for (char c : result) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (!prevWasSpace) {
                finalResult += ' ';
                prevWasSpace = true;
            }
        } else {
            finalResult += c;
            prevWasSpace = false;
        }
    }
    
    return finalResult;
}

std::string TextPreprocessor::removeStopwords(const std::string& text) {
    std::stringstream ss(text);
    std::string word;
    std::string result;
    
    while (ss >> word) {
        if (stops.find(word) == stops.end()) {
            if (!result.empty()) {
                result += " ";
            }
            result += word;
        }
    }
    
    return result;
}

std::string TextPreprocessor::stemWords(const std::string& text) {
    // This would require a Porter stemmer or similar library
    // For now, just return the original text
    return text;
}

} // namespace nlp
/**
 * @file tfidf_vectorizer.cpp
 * @brief Implementation of the TfidfVectorizer class
 * 
 * This file contains the implementation for converting text to TF-IDF
 * (Term Frequency-Inverse Document Frequency) numerical features,
 * which quantifies the importance of words in documents.
 */

#include "sentiment_analysis.h"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <set>
#include <numeric>

namespace nlp {

/**
 * Constructor initializes the vectorizer with configuration parameters.
 * 
 * @param sublinearTf Whether to apply sublinear scaling to term frequencies
 * @param maxDf Maximum document frequency cutoff for terms
 * @param maxFeatures Maximum number of features to extract
 */
TfidfVectorizer::TfidfVectorizer(bool sublinearTf, double maxDf, size_t maxFeatures)
    : sublinearTf(sublinearTf), maxDf(maxDf), maxFeatures(maxFeatures), documentCount(0) {
}

/**
 * Tokenizes text into words using a functional approach.
 * 
 * This function demonstrates how to break a string into tokens (words)
 * and is a fundamental operation for text processing.
 * 
 * @param text The input text to tokenize
 * @return Vector of tokens (words)
 */
std::vector<std::string> TfidfVectorizer::tokenize(std::string_view text) const {
    // Simple tokenization approach
    std::vector<std::string> tokens;
    std::istringstream iss{std::string{text}};
    std::string token;
    
    // Extract tokens one by one
    while (iss >> token) {
        tokens.push_back(token);
    }
    
    return tokens;
}

/**
 * Builds vocabulary from a corpus of texts.
 * 
 * This function analyzes a collection of texts to identify unique terms
 * and build a vocabulary, filtering by document frequency and limiting
 * to the top terms by frequency.
 * 
 * @param texts Vector of text documents
 */
void TfidfVectorizer::buildVocabulary(const std::vector<std::string_view>& texts) {
    // Reset state
    vocabulary.clear();
    documentFrequencies.clear();
    documentCount = texts.size();
    
    // Process document frequencies
    for (const auto& text : texts) {
        // Get unique tokens in this document
        auto tokens = tokenize(text);
        std::set<std::string> uniqueTokens(tokens.begin(), tokens.end());
        
        // Increment document frequency for each unique token
        for (const auto& token : uniqueTokens) {
            documentFrequencies[token]++;
        }
    }
    
    // Filter terms by document frequency and prepare for sorting
    std::vector<std::pair<std::string, size_t>> termFrequencies;
    
    // Transform document frequencies to term frequencies
    for (const auto& [term, freq] : documentFrequencies) {
        double docFreqRatio = static_cast<double>(freq) / documentCount;
        
        // Only include terms that don't appear in too many documents
        if (docFreqRatio <= maxDf) {
            termFrequencies.emplace_back(term, freq);
        }
    }
    
    // Sort by document frequency (descending)
    auto compareFreq = [](const auto& a, const auto& b) {
        return a.second > b.second;
    };
    
    std::ranges::sort(termFrequencies, compareFreq);
    
    // Limit to maxFeatures
    if (termFrequencies.size() > maxFeatures) {
        termFrequencies.resize(maxFeatures);
    }
    
    // Build vocabulary with indices
    size_t idx = 0;
    for (const auto& [term, freq] : termFrequencies) {
        vocabulary[term] = idx++;
    }
}

/**
 * Computes TF-IDF scores for a document.
 * 
 * This function calculates the Term Frequency-Inverse Document Frequency
 * scores for each term in a document, which quantifies how important
 * a word is to a document in a corpus.
 * 
 * @param text The input text document
 * @return Map of terms to their TF-IDF scores
 */
std::unordered_map<std::string, double> TfidfVectorizer::computeTfIdf(std::string_view text) const {
    std::unordered_map<std::string, double> result;
    
    // Skip empty text or empty vocabulary
    if (text.empty() || vocabulary.empty()) {
        return result;
    }
    
    // Count term frequencies in this document
    std::unordered_map<std::string, size_t> termCounts;
    auto tokens = tokenize(text);
    
    for (const auto& token : tokens) {
        // Only count tokens in our vocabulary
        if (vocabulary.find(token) != vocabulary.end()) {
            termCounts[token]++;
        }
    }
    
    // Compute TF-IDF for each term in the vocabulary
    for (const auto& [term, count] : termCounts) {
        // Calculate term frequency
        double tf = static_cast<double>(count);
        
        // Apply sublinear scaling if requested
        if (sublinearTf) {
            tf = 1.0 + std::log(tf);
        }
        
        // Get document frequency
        auto dfIt = documentFrequencies.find(term);
        double df = (dfIt != documentFrequencies.end()) ? static_cast<double>(dfIt->second) : 1.0;
        
        // Calculate inverse document frequency with smoothing
        double idf = std::log(static_cast<double>(documentCount) / (1.0 + df)) + 1.0;
        
        // Store TF-IDF score
        result[term] = tf * idf;
    }
    
    return result;
}

} // namespace nlp

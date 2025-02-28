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
    // Recursive approach to tokenization
    const auto tokenizeHelper = [](std::istringstream& ss, std::vector<std::string>& tokens) -> void {
        std::string token;
        
        // Base case: no more tokens to extract
        if (!(ss >> token)) {
            return;
        }
        
        // Add token and recursively process the rest
        tokens.push_back(token);
        tokenizeHelper(ss, tokens);
    };
    
    std::vector<std::string> tokens;
    std::istringstream ss{std::string{text}};
    tokenizeHelper(ss, tokens);
    
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
    
    // Process document frequencies using recursion
    const auto processTexts = [this](auto& self, const auto& texts, size_t index) -> void {
        // Base case: all texts processed
        if (index >= texts.size()) {
            return;
        }
        
        // Get unique tokens in this document
        auto tokens = tokenize(texts[index]);
        std::set<std::string> uniqueTokens(tokens.begin(), tokens.end());
        
        // Increment document frequency for each unique token
        for (const auto& token : uniqueTokens) {
            documentFrequencies[token]++;
        }
        
        // Recursively process the next document
        self(self, texts, index + 1);
    };
    
    // Start the recursive processing
    processTexts(processTexts, texts, 0);
    
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
    
    // Count term frequencies in this document using recursion
    const auto countTerms = [](auto& self, const auto& tokens, size_t index, 
                              std::unordered_map<std::string, size_t>& counts) -> void {
        // Base case: all tokens processed
        if (index >= tokens.size()) {
            return;
        }
        
        // Increment count for this token
        counts[tokens[index]]++;
        
        // Recursively process the next token
        self(self, tokens, index + 1, counts);
    };
    
    // Get tokens and count frequencies
    auto tokens = tokenize(text);
    std::unordered_map<std::string, size_t> termCounts;
    countTerms(countTerms, tokens, 0, termCounts);
    
    // Compute TF-IDF for each term in the vocabulary
    const auto computeScores = [this](auto& self, const auto& termCounts, 
                                   auto iter, std::unordered_map<std::string, double>& scores) -> void {
        // Base case: all terms processed
        if (iter == termCounts.end()) {
            return;
        }
        
        const auto& [term, count] = *iter;
        
        // Skip terms not in our vocabulary
        if (!vocabulary.contains(term)) {
            self(self, termCounts, std::next(iter), scores);
            return;
        }
        
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
        scores[term] = tf * idf;
        
        // Recursively process the next term
        self(self, termCounts, std::next(iter), scores);
    };
    
    // Start the recursive computation
    computeScores(computeScores, termCounts, termCounts.begin(), result);
    
    return result;
}

} // namespace nlp

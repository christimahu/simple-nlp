/**
 * @file tfidf_vectorizer.cpp
 * @brief Implements the TF-IDF vectorization class for text feature extraction.
 *
 * This file provides the implementation for the TF-IDF vectorizer, which
 * converts text documents into numerical feature vectors based on word
 * importance. TF-IDF helps capture the significance of words in documents
 * relative to a corpus by balancing term frequency with inverse document
 * frequency.
 */

#include "tfidf_vectorizer.h"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <utility>
#include <iostream>
#include <set>

namespace nlp {

/**
 * @brief Constructs a TF-IDF vectorizer with the specified parameters.
 * 
 * @param sublinearTf If true, applies logarithmic scaling to term frequencies.
 * @param maxDf Maximum document frequency threshold for vocabulary filtering.
 * @param maxFeatures Maximum number of features to retain in the vocabulary.
 */
TfidfVectorizer::TfidfVectorizer(bool sublinearTf, double maxDf, size_t maxFeatures)
    : sublinearTf(sublinearTf), 
      maxDf(maxDf), 
      maxFeatures(maxFeatures),
      totalDocuments(0) {}

/**
 * @brief Tokenizes text into words by splitting on whitespace.
 * 
 * This method converts a text string into a sequence of tokens (words)
 * by splitting on whitespace characters. It does not perform advanced
 * tokenization like handling contractions or special characters.
 * 
 * @param text The input text to tokenize.
 * @return A vector of tokens extracted from the text.
 */
std::vector<std::string> TfidfVectorizer::tokenize(std::string_view text) const {
    std::istringstream iss{std::string(text)};
    std::vector<std::string> tokens;
    std::string word;

    while (iss >> word) {
        if (!word.empty()) {
            tokens.push_back(word);
        }
    }
    
    return tokens;
}

/**
 * @brief Fits the vectorizer to a corpus of documents.
 * 
 * This method analyzes the input corpus to build a vocabulary and
 * calculate document frequencies, which are needed for TF-IDF computation.
 * 
 * @param texts Collection of text documents to analyze.
 * @param maxDf Maximum document frequency threshold (0.0-1.0 or absolute count).
 * @param maxFeatures Maximum number of features to keep.
 */
void TfidfVectorizer::fit(const std::vector<std::string>& texts, 
                          double maxDf, 
                          size_t maxFeatures) {
    // Reset state
    vocabulary.clear();
    documentFrequencies.clear();
    totalDocuments = texts.size();
    
    if (totalDocuments == 0) {
        return;  // Nothing to fit
    }
    
    // Tokenize all documents
    std::vector<std::vector<std::string>> tokenizedDocs;
    tokenizedDocs.reserve(texts.size());
    
    for (const auto& text : texts) {
        tokenizedDocs.push_back(tokenize(text));
    }
    
    // Build vocabulary and calculate document frequencies
    buildVocabulary(tokenizedDocs, maxDf, maxFeatures);
    
    std::cout << "Built vocabulary with " << vocabulary.size() << " features" << std::endl;
}

/**
 * @brief Transforms documents into TF-IDF feature vectors.
 * 
 * This method converts each document into a TF-IDF feature vector using
 * the vocabulary and document frequencies from a previous fit() call.
 * 
 * @param texts Collection of documents to transform.
 * @return Matrix of TF-IDF features (one vector per document).
 */
std::vector<std::vector<double>> TfidfVectorizer::transform(
    const std::vector<std::string>& texts) const {
    
    if (vocabulary.empty()) {
        throw std::runtime_error("Vocabulary is empty. Call fit() first.");
    }
    
    std::vector<std::vector<double>> featureMatrix;
    featureMatrix.reserve(texts.size());
    
    // Process each document
    for (const auto& text : texts) {
        // Count term frequencies in this document
        std::unordered_map<std::string, int> termFreqs;
        for (const auto& token : tokenize(text)) {
            termFreqs[token]++;
        }
        
        // Convert to TF-IDF vector
        std::vector<double> featureVector(vocabulary.size(), 0.0);
        
        for (const auto& [term, freq] : termFreqs) {
            // Skip terms not in vocabulary
            auto it = vocabulary.find(term);
            if (it == vocabulary.end()) {
                continue;
            }
            
            // Get the feature index and document frequency
            int featureIdx = it->second;
            int docFreq = documentFrequencies[featureIdx];
            
            // Calculate TF-IDF
            double tf = static_cast<double>(freq);
            if (sublinearTf) {
                tf = 1.0 + std::log(tf);  // Sublinear scaling
            }
            
            double idf = std::log(static_cast<double>(totalDocuments + 1) / 
                                 (docFreq + 1)) + 1.0;  // Smoothed IDF
            
            featureVector[featureIdx] = tf * idf;
        }
        
        // Normalize the feature vector (L2 norm)
        double squaredSum = 0.0;
        for (double val : featureVector) {
            squaredSum += val * val;
        }
        
        if (squaredSum > 0.0) {
            double norm = std::sqrt(squaredSum);
            for (double& val : featureVector) {
                val /= norm;
            }
        }
        
        featureMatrix.push_back(std::move(featureVector));
    }
    
    return featureMatrix;
}

/**
 * @brief Fits the vectorizer to data and transforms it in one step.
 * 
 * This convenience method combines the fit() and transform() steps.
 * 
 * @param texts Collection of documents to analyze and transform.
 * @return Matrix of TF-IDF features.
 */
std::vector<std::vector<double>> TfidfVectorizer::fitTransform(
    const std::vector<std::string>& texts) {
    
    fit(texts, maxDf, maxFeatures);
    return transform(texts);
}

/**
 * @brief Builds vocabulary from tokenized documents.
 * 
 * This private method analyzes term frequencies across the corpus
 * to construct a vocabulary and calculate document frequencies.
 * 
 * @param tokenizedDocs Collection of tokenized documents.
 * @param maxDf Maximum document frequency threshold.
 * @param maxFeatures Maximum vocabulary size.
 */
void TfidfVectorizer::buildVocabulary(
    const std::vector<std::vector<std::string>>& tokenizedDocs,
    double maxDf,
    size_t maxFeatures) {
    
    // Count document frequency for each term
    std::unordered_map<std::string, int> docFreqs;
    
    for (const auto& doc : tokenizedDocs) {
        // Use set to count each term only once per document
        std::set<std::string> uniqueTerms(doc.begin(), doc.end());
        
        for (const auto& term : uniqueTerms) {
            docFreqs[term]++;
        }
    }
    
    // Convert maxDf from fraction to absolute count if needed
    int maxDfCount = (maxDf >= 1.0) ? 
                     static_cast<int>(maxDf) : 
                     static_cast<int>(maxDf * totalDocuments);
    
    // Filter terms by document frequency
    std::vector<std::pair<std::string, int>> filteredTerms;
    for (const auto& [term, freq] : docFreqs) {
        if (freq <= maxDfCount) {
            filteredTerms.emplace_back(term, freq);
        }
    }
    
    // Sort by descending frequency for feature selection
    std::sort(filteredTerms.begin(), filteredTerms.end(),
             [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Limit to maxFeatures
    if (filteredTerms.size() > maxFeatures) {
        filteredTerms.resize(maxFeatures);
    }
    
    // Build vocabulary and document frequency vector
    vocabulary.clear();
    documentFrequencies.clear();
    documentFrequencies.reserve(filteredTerms.size());
    
    for (size_t i = 0; i < filteredTerms.size(); ++i) {
        const auto& [term, freq] = filteredTerms[i];
        vocabulary[term] = static_cast<int>(i);
        documentFrequencies.push_back(freq);
    }
}

/**
 * @brief Calculates TF-IDF score for a term in a document.
 * 
 * This helper method computes the TF-IDF score for a single term
 * based on its frequency in the document and across the corpus.
 * 
 * @param termFreq Frequency of the term in the document.
 * @param docFreq Number of documents containing the term.
 * @param totalDocs Total number of documents in the corpus.
 * @return TF-IDF score.
 */
double TfidfVectorizer::calculateTfIdf(int termFreq, int docFreq, int totalDocs) const {
    double tf = sublinearTf ? 1.0 + std::log(termFreq) : static_cast<double>(termFreq);
    double idf = std::log(static_cast<double>(totalDocs + 1) / (docFreq + 1)) + 1.0;
    
    return tf * idf;
}

} // namespace nlp

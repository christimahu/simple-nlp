#include "sentiment_analysis.h"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <set>

namespace nlp {

TfidfVectorizer::TfidfVectorizer(bool sublinearTf, double maxDf, size_t maxFeatures)
    : sublinearTf(sublinearTf), maxDf(maxDf), maxFeatures(maxFeatures), documentCount(0) {
}

std::vector<std::string> TfidfVectorizer::tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::stringstream ss(text);
    std::string token;
    
    while (ss >> token) {
        tokens.push_back(token);
    }
    
    return tokens;
}

void TfidfVectorizer::buildVocabulary(const std::vector<std::string>& texts) {
    documentCount = texts.size();
    std::unordered_map<std::string, size_t> termFreq;
    
    // Count term frequencies across all documents
    for (const auto& text : texts) {
        std::vector<std::string> tokens = tokenize(text);
        std::set<std::string> uniqueTokens(tokens.begin(), tokens.end());
        
        for (const auto& token : uniqueTokens) {
            documentFrequencies[token]++;
        }
        
        for (const auto& token : tokens) {
            termFreq[token]++;
        }
    }
    
    // Filter terms by document frequency
    std::vector<std::pair<std::string, size_t>> sortedTerms;
    for (const auto& [term, freq] : termFreq) {
        double docFreqRatio = static_cast<double>(documentFrequencies[term]) / documentCount;
        if (docFreqRatio <= maxDf) {
            sortedTerms.push_back({term, freq});
        }
    }
    
    // Sort by term frequency (descending)
    std::sort(sortedTerms.begin(), sortedTerms.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Limit to max_features
    if (sortedTerms.size() > maxFeatures) {
        sortedTerms.resize(maxFeatures);
    }
    
    // Build vocabulary
    size_t idx = 0;
    for (const auto& [term, freq] : sortedTerms) {
        vocabulary[term] = idx++;
    }
}

std::unordered_map<std::string, double> TfidfVectorizer::computeTfIdf(const std::string& text) {
    std::unordered_map<std::string, double> result;
    std::vector<std::string> tokens = tokenize(text);
    
    // Count term frequencies in this document
    std::unordered_map<std::string, size_t> termCounts;
    for (const auto& token : tokens) {
        if (vocabulary.find(token) != vocabulary.end()) {
            termCounts[token]++;
        }
    }
    
    // Compute TF-IDF for each term
    for (const auto& [term, count] : termCounts) {
        double tf = static_cast<double>(count);
        if (sublinearTf) {
            tf = 1.0 + std::log(tf);
        }
        
        double idf = std::log(static_cast<double>(documentCount) / 
                            (1.0 + static_cast<double>(documentFrequencies[term]))) + 1.0;
        
        result[term] = tf * idf;
    }
    
    return result;
}

std::vector<std::vector<double>> TfidfVectorizer::fitTransform(const std::vector<std::string>& texts) {
    // Build vocabulary first
    buildVocabulary(texts);
    
    // Then transform the texts
    return transform(texts);
}

std::vector<std::vector<double>> TfidfVectorizer::transform(const std::vector<std::string>& texts) {
    std::vector<std::vector<double>> result(texts.size(), std::vector<double>(vocabulary.size(), 0.0));
    
    for (size_t i = 0; i < texts.size(); ++i) {
        std::unordered_map<std::string, double> tfidfScores = computeTfIdf(texts[i]);
        
        for (const auto& [term, score] : tfidfScores) {
            auto it = vocabulary.find(term);
            if (it != vocabulary.end()) {
                result[i][it->second] = score;
            }
        }
    }
    
    return result;
}

} // namespace nlp
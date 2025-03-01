/**
 * @file sentiment_dataset.h
 * @brief Defines the SentimentDataset class for managing sentiment analysis data.
 */

#pragma once

#include <vector>
#include <string>
#include <utility>
#include <optional>

namespace nlp {

/**
 * @class SentimentDataset
 * @brief Manages data for sentiment analysis, including text and labels.
 * 
 * This class handles loading, preprocessing, and splitting of sentiment analysis data.
 * It uses a functional programming approach with recursion instead of loops where appropriate.
 */
class SentimentDataset {
public:
    /**
     * @brief Default constructor
     */
    SentimentDataset() = default;
    
    /**
     * @brief Constructor with data
     * @param data Vector of (sentiment_label, text) pairs
     */
    explicit SentimentDataset(std::vector<std::pair<int, std::string>> data);
    
    /**
     * @brief Get training texts based on train indices
     * @return Vector of training text samples
     */
    std::vector<std::string> getTrainTexts() const;
    
    /**
     * @brief Get test texts based on test indices
     * @return Vector of test text samples
     */
    std::vector<std::string> getTestTexts() const;
    
    /**
     * @brief Get training labels based on train indices
     * @return Vector of training labels
     */
    std::vector<int> getTrainLabels() const;
    
    /**
     * @brief Get test labels based on test indices
     * @return Vector of test labels
     */
    std::vector<int> getTestLabels() const;
    
    /**
     * @brief Get filtered texts with a specific sentiment
     * @param sentiment Sentiment value (0=negative, 4=positive)
     * @return Vector of texts with the specified sentiment
     */
    std::vector<std::string> getTextsWithSentiment(int sentiment) const;
    
    /**
     * @brief Get training feature vectors
     * @return Matrix of training features
     */
    std::vector<std::vector<double>> getTrainFeatures() const;
    
    /**
     * @brief Get test feature vectors
     * @return Matrix of test features
     */
    std::vector<std::vector<double>> getTestFeatures() const;

public:
    // Raw data from source
    std::vector<std::pair<int, std::string>> data;
    
    // Preprocessed data
    std::vector<std::string> cleanedTexts;
    
    // Labels extracted from data
    std::vector<int> labels;
    
    // Feature vectors extracted from texts
    std::vector<std::vector<double>> features;
    
    // Indices for train/test split
    std::vector<size_t> trainIndices;
    std::vector<size_t> testIndices;
};

} // namespace nlp

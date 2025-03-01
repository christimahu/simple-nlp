/**
 * @file sentiment_dataset.cpp
 * @brief Implementation of the SentimentDataset class.
 * 
 * This file contains the implementation of methods for the SentimentDataset class,
 * which handles data management and organization for sentiment analysis. It follows
 * a functional programming approach with recursion instead of traditional loops.
 */

#include "sentiment_dataset.h"
#include <algorithm>
#include <functional>
#include <sstream>

namespace nlp {

/**
 * @brief Constructor with initial data.
 * 
 * Initializes a dataset with sentiment-text pairs.
 * 
 * @param data Vector of (sentiment_label, text) pairs
 */
SentimentDataset::SentimentDataset(std::vector<std::pair<int, std::string>> data)
    : data(std::move(data)) {
    
    // Extract labels
    labels.reserve(this->data.size());
    for (const auto& [sentiment, _] : this->data) {
        labels.push_back(sentiment);
    }
}

/**
 * @brief Get training texts based on train indices.
 * 
 * This method uses recursion to extract training texts from the dataset
 * based on the train indices.
 * 
 * @return Vector of training texts
 */
std::vector<std::string> SentimentDataset::getTrainTexts() const {
    std::vector<std::string> trainTexts;
    trainTexts.reserve(trainIndices.size());
    
    // Extract texts recursively
    const auto extractTexts = [this, &trainTexts](auto& self, size_t index) -> void {
        // Base case: all indices processed
        if (index >= trainIndices.size()) {
            return;
        }
        
        // Add this text
        size_t dataIndex = trainIndices[index];
        if (dataIndex < cleanedTexts.size()) {
            trainTexts.push_back(cleanedTexts[dataIndex]);
        }
        
        // Recursively process next index
        self(self, index + 1);
    };
    
    extractTexts(extractTexts, 0);
    
    return trainTexts;
}

/**
 * @brief Get test texts based on test indices.
 * 
 * This method uses recursion to extract test texts from the dataset
 * based on the test indices.
 * 
 * @return Vector of test texts
 */
std::vector<std::string> SentimentDataset::getTestTexts() const {
    std::vector<std::string> testTexts;
    testTexts.reserve(testIndices.size());
    
    // Extract texts recursively
    const auto extractTexts = [this, &testTexts](auto& self, size_t index) -> void {
        // Base case: all indices processed
        if (index >= testIndices.size()) {
            return;
        }
        
        // Add this text
        size_t dataIndex = testIndices[index];
        if (dataIndex < cleanedTexts.size()) {
            testTexts.push_back(cleanedTexts[dataIndex]);
        }
        
        // Recursively process next index
        self(self, index + 1);
    };
    
    extractTexts(extractTexts, 0);
    
    return testTexts;
}

/**
 * @brief Get training labels based on train indices.
 * 
 * This method uses recursion to extract training labels from the dataset
 * based on the train indices.
 * 
 * @return Vector of training labels
 */
std::vector<int> SentimentDataset::getTrainLabels() const {
    std::vector<int> trainLabels;
    trainLabels.reserve(trainIndices.size());
    
    // Extract labels recursively
    const auto extractLabels = [this, &trainLabels](auto& self, size_t index) -> void {
        // Base case: all indices processed
        if (index >= trainIndices.size()) {
            return;
        }
        
        // Add this label
        size_t dataIndex = trainIndices[index];
        if (dataIndex < labels.size()) {
            trainLabels.push_back(labels[dataIndex]);
        }
        
        // Recursively process next index
        self(self, index + 1);
    };
    
    extractLabels(extractLabels, 0);
    
    return trainLabels;
}

/**
 * @brief Get test labels based on test indices.
 * 
 * This method uses recursion to extract test labels from the dataset
 * based on the test indices.
 * 
 * @return Vector of test labels
 */
std::vector<int> SentimentDataset::getTestLabels() const {
    std::vector<int> testLabels;
    testLabels.reserve(testIndices.size());
    
    // Extract labels recursively
    const auto extractLabels = [this, &testLabels](auto& self, size_t index) -> void {
        // Base case: all indices processed
        if (index >= testIndices.size()) {
            return;
        }
        
        // Add this label
        size_t dataIndex = testIndices[index];
        if (dataIndex < labels.size()) {
            testLabels.push_back(labels[dataIndex]);
        }
        
        // Recursively process next index
        self(self, index + 1);
    };
    
    extractLabels(extractLabels, 0);
    
    return testLabels;
}

/**
 * @brief Get texts with a specific sentiment from the dataset.
 * 
 * This method uses recursion to filter texts with a particular sentiment
 * label from the dataset.
 * 
 * @param sentiment Sentiment value (0=negative, 4=positive)
 * @return Vector of texts with the specified sentiment
 */
std::vector<std::string> SentimentDataset::getTextsWithSentiment(int sentiment) const {
    std::vector<std::string> filteredTexts;
    
    // Filter texts recursively
    const auto filterTexts = [this, sentiment, &filteredTexts](auto& self, size_t index) -> void {
        // Base case: all texts checked
        if (index >= data.size() || index >= cleanedTexts.size()) {
            return;
        }
        
        // Check sentiment and add text if it matches
        if (data[index].first == sentiment) {
            filteredTexts.push_back(cleanedTexts[index]);
        }
        
        // Recursively check next text
        self(self, index + 1);
    };
    
    filterTexts(filterTexts, 0);
    
    return filteredTexts;
}

/**
 * @brief Get feature vectors for training data.
 * 
 * This method retrieves feature vectors for training examples
 * using the train indices.
 * 
 * @return Matrix of training feature vectors
 */
std::vector<std::vector<double>> SentimentDataset::getTrainFeatures() const {
    std::vector<std::vector<double>> trainFeatures;
    trainFeatures.reserve(trainIndices.size());
    
    // Extract features recursively
    const auto extractFeatures = [this, &trainFeatures](auto& self, size_t index) -> void {
        // Base case: all indices processed
        if (index >= trainIndices.size() || features.empty()) {
            return;
        }
        
        // Add this feature vector
        size_t dataIndex = trainIndices[index];
        if (dataIndex < features.size()) {
            trainFeatures.push_back(features[dataIndex]);
        }
        
        // Recursively process next index
        self(self, index + 1);
    };
    
    extractFeatures(extractFeatures, 0);
    
    return trainFeatures;
}

/**
 * @brief Get feature vectors for test data.
 * 
 * This method retrieves feature vectors for test examples
 * using the test indices.
 * 
 * @return Matrix of test feature vectors
 */
std::vector<std::vector<double>> SentimentDataset::getTestFeatures() const {
    std::vector<std::vector<double>> testFeatures;
    testFeatures.reserve(testIndices.size());
    
    // Extract features recursively
    const auto extractFeatures = [this, &testFeatures](auto& self, size_t index) -> void {
        // Base case: all indices processed
        if (index >= testIndices.size() || features.empty()) {
            return;
        }
        
        // Add this feature vector
        size_t dataIndex = testIndices[index];
        if (dataIndex < features.size()) {
            testFeatures.push_back(features[dataIndex]);
        }
        
        // Recursively process next index
        self(self, index + 1);
    };
    
    extractFeatures(extractFeatures, 0);
    
    return testFeatures;
}

} // namespace nlp

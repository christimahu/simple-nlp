/**
 * @file sentiment_dataset.cpp
 * @brief Implementation of the SentimentDataset class
 * 
 * This file contains the implementation of methods for the SentimentDataset class,
 * which handles data management and organization for sentiment analysis. It uses
 * a functional programming approach with recursion instead of traditional loops.
 */

#include "sentiment_analysis.h"
#include <algorithm>
#include <functional>

namespace nlp {

/**
 * Get training texts based on trainIndices.
 * 
 * This method demonstrates dataset manipulation and subset extraction
 * using recursion instead of loops.
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
 * Get test texts based on testIndices.
 * 
 * This method demonstrates dataset manipulation and subset extraction
 * using recursion instead of loops.
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
 * Get training labels based on trainIndices.
 * 
 * This method demonstrates dataset manipulation and subset extraction
 * using recursion instead of loops.
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
 * Get test labels based on testIndices.
 * 
 * This method demonstrates dataset manipulation and subset extraction
 * using recursion instead of loops.
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
 * Get texts with a specific sentiment from the dataset.
 * 
 * This method demonstrates filtering operation using recursion
 * to select texts with a particular sentiment label.
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
 * Get feature vectors for training data.
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
 * Get feature vectors for test data.
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

/**
 * Calculate class distribution in the dataset.
 * 
 * This method counts the occurrences of each sentiment class
 * in the dataset, providing insights into data balance.
 * 
 * @return Map of sentiment values to their frequencies
 */
std::unordered_map<int, size_t> SentimentDataset::getClassDistribution() const {
    std::unordered_map<int, size_t> distribution;
    
    // Count classes recursively
    const auto countClasses = [this, &distribution](auto& self, size_t index) -> void {
        // Base case: all labels counted
        if (index >= labels.size()) {
            return;
        }
        
        // Count this label
        distribution[labels[index]]++;
        
        // Recursively count next label
        self(self, index + 1);
    };
    
    countClasses(countClasses, 0);
    
    return distribution;
}

/**
 * Get statistics about the dataset.
 * 
 * This method provides basic statistics about the dataset,
 * such as its size, class distribution, and unique words.
 * 
 * @return Map of statistic names to their values
 */
std::map<std::string, std::string> SentimentDataset::getStatistics() const {
    std::map<std::string, std::string> stats;
    
    // Basic size information
    stats["total_samples"] = std::to_string(data.size());
    stats["cleaned_samples"] = std::to_string(cleanedTexts.size());
    stats["train_samples"] = std::to_string(trainIndices.size());
    stats["test_samples"] = std::to_string(testIndices.size());
    
    // Get class distribution
    auto distribution = getClassDistribution();
    
    // Format class distribution as a string
    std::stringstream distStream;
    
    // Add distribution entries recursively
    const auto addDistribution = [&distStream](auto& self, auto it, auto end, bool isFirst) -> void {
        // Base case: all entries added
        if (it == end) {
            return;
        }
        
        // Add separator if needed
        if (!isFirst) {
            distStream << ", ";
        }
        
        // Add this entry
        distStream << "Class " << it->first << ": " << it->second;
        
        // Recursively add next entry
        self(self, std::next(it), end, false);
    };
    
    addDistribution(addDistribution, distribution.begin(), distribution.end(), true);
    stats["class_distribution"] = distStream.str();
    
    // Count unique words in the dataset
    std::set<std::string> uniqueWords;
    
    // Process each text recursively
    const auto processTexts = [this, &uniqueWords](auto& self, size_t textIndex) -> void {
        // Base case: all texts processed
        if (textIndex >= cleanedTexts.size()) {
            return;
        }
        
        // Get words from this text
        std::istringstream iss(cleanedTexts[textIndex]);
        
        // Process each word recursively
        const auto processWords = [&uniqueWords](auto& self, std::istringstream& iss) -> void {
            std::string word;
            
            // Base case: no more words to extract
            if (!(iss >> word)) {
                return;
            }
            
            // Add this word to the set
            uniqueWords.insert(word);
            
            // Recursively process next word
            self(self, iss);
        };
        
        processWords(processWords, iss);
        
        // Recursively process next text
        self(self, textIndex + 1);
    };
    
    processTexts(processTexts, 0);
    
    stats["unique_words"] = std::to_string(uniqueWords.size());
    
    return stats;
}

} // namespace nlp

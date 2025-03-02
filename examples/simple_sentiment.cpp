/**
 * @file simple_sentiment.cpp
 * @brief Example application showing basic sentiment analysis usage
 * 
 * This example demonstrates how to use the NLP sentiment analysis library
 * for a simple sentiment classification task. It shows the complete
 * pipeline from data preparation to model training and prediction.
 * 
 * The example uses a small dataset of reviews with positive and negative
 * sentiment labels, preprocesses the text, extracts features, trains a model,
 * and makes predictions on new examples.
 */

#include "sentiment_analysis.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <memory>

/**
 * Function to demonstrate sentiment analysis on a dataset of reviews.
 * 
 * This function shows how to:
 * - Create and preprocess a sentiment dataset
 * - Extract TF-IDF features from text
 * - Split data into training and testing sets
 * - Train a sentiment classification model
 * - Evaluate model performance
 * - Make predictions on new examples
 */
void runSentimentDemo() {
    std::cout << "==== NLP Sentiment Analysis Demo ====" << std::endl;
    
    // Create a more substantial sample dataset for better training
    std::vector<std::pair<int, std::string>> sampleData = {
        {4, "This product is amazing! I love it so much."},
        {4, "Great experience, would definitely recommend."},
        {4, "The service was excellent and staff very friendly."},
        {4, "I'm completely satisfied with my purchase."},
        {4, "Best customer support I've ever encountered."},
        {4, "Really happy with the quality of this product."},
        {4, "Works exactly as described and exceeds expectations."},
        {4, "Shipping was fast and the item arrived in perfect condition."},
        {4, "The value for money is incredible with this purchase."},
        {4, "The product design is beautiful and functional."},
        {4, "Customer service responded quickly and solved my issue."},
        {4, "Very intuitive and easy to use, highly recommend."},
        {0, "Terrible product, broke after first use."},
        {0, "Very disappointed with the quality."},
        {0, "Customer service was awful and unhelpful."},
        {0, "Would not recommend to anyone."},
        {0, "Worst purchase I've ever made."},
        {0, "The product didn't work as advertised."},
        {0, "Completely frustrated with this experience."},
        {0, "Poor design and even worse functionality."},
        {0, "Save your money and avoid this product."},
        {0, "The company doesn't stand behind their product."},
        {0, "Shipping took forever and the item arrived damaged."},
        {0, "Way overpriced for the quality you receive."}
    };
    
    // Initialize sentiment analyzer
    nlp::SentimentAnalyzer analyzer;
    
    // Create dataset
    nlp::SentimentDataset dataset(sampleData);
    
    // Preprocess data
    std::cout << "\nPreprocessing text data..." << std::endl;
    dataset = analyzer.preprocessData(std::move(dataset));
    
    std::cout << "Preprocessed " << dataset.cleanedTexts.size() << " texts" << std::endl;
    
    // Show sample of preprocessed data for demonstration
    const auto showSamples = [&](auto& self, size_t index, size_t maxSamples) -> void {
        if (index >= dataset.data.size() || index >= maxSamples) {
            return;
        }
        
        std::cout << "Original: " << dataset.data[index].second << std::endl;
        std::cout << "Cleaned: " << dataset.cleanedTexts[index] << std::endl;
        std::cout << "Sentiment: " << (dataset.data[index].first == 4 ? "Positive" : "Negative") << std::endl;
        std::cout << std::endl;
        
        self(self, index + 1, maxSamples);
    };
    
    showSamples(showSamples, 0, 3);
    
    // Extract features with reduced dimensionality to avoid overfitting
    std::cout << "Extracting TF-IDF features..." << std::endl;
    auto featurePair = analyzer.extractFeatures(dataset, 0.9, 25);
    
    // Store features in dataset
    dataset.features = featurePair.first;
    dataset.labels = featurePair.second;
    
    std::cout << "Extracted " << dataset.features[0].size() << " features from " 
              << dataset.features.size() << " texts" << std::endl;
    
    // Split data using 80% for training and 20% for testing
    std::cout << "Splitting data into train/test sets..." << std::endl;
    auto splitDataset = analyzer.splitData(std::move(dataset), 0.2);
    
    // Train model with stronger regularization
    std::cout << "Training sentiment classifier..." << std::endl;
    auto X_train = splitDataset.getTrainFeatures();
    auto y_train = splitDataset.getTrainLabels();
    
    auto startTime = std::chrono::high_resolution_clock::now();
    // Use stronger regularization (alpha=0.01) to prevent overfitting
    auto model = analyzer.trainModel(X_train, y_train, 0.01, 0.01);
    auto endTime = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "Model trained in " << duration.count() << " ms" << std::endl;
    
    // Evaluate model
    std::cout << "\nEvaluating model performance..." << std::endl;
    auto X_test = splitDataset.getTestFeatures();
    auto y_test = splitDataset.getTestLabels();
    auto metrics = analyzer.evaluateModel(*model, X_test, y_test);
    
    std::cout << "Accuracy: " << std::fixed << std::setprecision(2) 
              << metrics["accuracy"] * 100.0 << "%" << std::endl;
    
    // Create vectorizer for new predictions
    nlp::TfidfVectorizer vectorizer(true, 0.9, 25);
    vectorizer.fitTransform(splitDataset.cleanedTexts);
    
    // Test with new examples of varying sentiment
    std::cout << "\nTesting with new examples:" << std::endl;
    
    std::vector<std::string> testExamples = {
        "This is absolutely fantastic! I would buy it again.",
        "I'm extremely disappointed with this. Complete waste of money.",
        "It's okay, nothing special but gets the job done.",
        "The customer service team was helpful and responsive.",
        "The product arrived late and was missing parts."
    };
    
    // Process each example recursively
    const auto processExamples = [&](auto& self, size_t index) -> void {
        if (index >= testExamples.size()) {
            return;
        }
        
        const std::string& text = testExamples[index];
        auto result = analyzer.predictSentiment(text, *model, vectorizer);
        
        std::cout << "Text: " << text << std::endl;
        std::cout << "Predicted sentiment: " << result.sentiment 
                  << " (confidence: " << std::fixed << std::setprecision(2) 
                  << result.confidence << ")" << std::endl;
        std::cout << "Explanation: " << result.explanation << std::endl;
        std::cout << std::endl;
        
        self(self, index + 1);
    };
    
    processExamples(processExamples, 0);
    
    std::cout << "\n==== Demo Complete ====" << std::endl;
}

/**
 * Main function for the example application.
 * 
 * Runs the sentiment analysis demonstration with error handling.
 * 
 * @return Exit status code (0 for success, 1 for error)
 */
int main() {
    try {
        runSentimentDemo();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

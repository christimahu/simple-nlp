/**
 * @file simple_sentiment.cpp
 * @brief Example application showing basic sentiment analysis usage
 * 
 * This example demonstrates how to use the NLP sentiment analysis library
 * for a simple sentiment classification task. It shows the complete
 * pipeline from loading data to model training and prediction.
 */

#include "sentiment_analysis.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <memory>

/**
 * Function to demonstrate sentiment analysis on a small dataset.
 * 
 * This function shows how to use the sentiment analysis library
 * with a simple dataset, training a model and making predictions.
 * It demonstrates a functional approach with recursion.
 */
void runSentimentDemo() {
    std::cout << "==== NLP Sentiment Analysis Demo ====" << std::endl;
    
    // Create sample dataset
    std::vector<std::pair<int, std::string>> sampleData = {
        {4, "This product is amazing! I love it so much."},
        {4, "Great experience, would definitely recommend."},
        {4, "The service was excellent and staff very friendly."},
        {4, "I'm completely satisfied with my purchase."},
        {4, "Best customer support I've ever encountered."},
        {0, "Terrible product, broke after first use."},
        {0, "Very disappointed with the quality."},
        {0, "Customer service was awful and unhelpful."},
        {0, "Would not recommend to anyone."},
        {0, "Worst purchase I've ever made."}
    };
    
    // Initialize sentiment analyzer
    nlp::SentimentAnalyzer analyzer;
    
    // Create dataset
    nlp::SentimentDataset dataset;
    dataset.data = sampleData;
    
    // Preprocess data
    std::cout << "\nPreprocessing text data..." << std::endl;
    // Use std::move to avoid copying the dataset with unique_ptr
    dataset = analyzer.preprocessData(std::move(dataset));
    
    // Show sample of preprocessed data using recursion
    const auto showSamples = [&](auto& self, size_t index, size_t maxSamples) -> void {
        // Base case: all samples shown or no more data
        if (index >= dataset.data.size() || index >= maxSamples) {
            return;
        }
        
        // Show this sample
        std::cout << "Original: " << dataset.data[index].second << std::endl;
        std::cout << "Cleaned: " << dataset.cleanedTexts[index] << std::endl;
        std::cout << "Sentiment: " << (dataset.data[index].first == 4 ? "Positive" : "Negative") << std::endl;
        std::cout << std::endl;
        
        // Recursively show next sample
        self(self, index + 1, maxSamples);
    };
    
    showSamples(showSamples, 0, 3);
    
    // Extract features
    std::cout << "Extracting TF-IDF features..." << std::endl;
    auto [features, labels] = analyzer.extractFeatures(dataset);
    
    // Split data into training and testing sets
    std::cout << "Splitting data into train/test sets..." << std::endl;
    dataset.features = features;
    dataset.labels = labels;
    // Use std::move to avoid copying the dataset with unique_ptr
    dataset = analyzer.splitData(std::move(dataset), 3); // 3 test samples
    
    // Train model
    std::cout << "Training sentiment classifier..." << std::endl;
    auto X_train = dataset.getTrainFeatures();
    auto y_train = dataset.getTrainLabels();
    
    auto startTime = std::chrono::high_resolution_clock::now();
    auto model = analyzer.trainModel(X_train, y_train);
    auto endTime = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "Model trained in " << duration.count() << " ms" << std::endl;
    
    // Evaluate model
    std::cout << "\nEvaluating model performance..." << std::endl;
    auto X_test = dataset.getTestFeatures();
    auto y_test = dataset.getTestLabels();
    auto metrics = analyzer.evaluateModel(*model, X_test, y_test);
    
    std::cout << "Accuracy: " << std::fixed << std::setprecision(2) 
              << metrics["accuracy"] * 100.0 << "%" << std::endl;
    
    // Create vectorizer for new predictions
    nlp::TfidfVectorizer vectorizer(true, 0.5, 100);
    vectorizer.fitTransform(dataset.cleanedTexts);
    
    // Test with new examples
    std::cout << "\nTesting with new examples:" << std::endl;
    
    // Define test examples
    std::vector<std::string> testExamples = {
        "This is absolutely fantastic!",
        "I'm extremely disappointed with this.",
        "It's okay, nothing special."
    };
    
    // Process each example recursively
    const auto processExamples = [&](auto& self, size_t index) -> void {
        // Base case: all examples processed
        if (index >= testExamples.size()) {
            return;
        }
        
        // Process this example
        const std::string& text = testExamples[index];
        auto result = analyzer.predictSentiment(text, *model, vectorizer);
        
        std::cout << "Text: " << text << std::endl;
        std::cout << "Predicted sentiment: " << result.sentiment 
                  << " (confidence: " << std::fixed << std::setprecision(2) 
                  << result.confidence << ")" << std::endl;
        std::cout << "Explanation: " << result.explanation << std::endl;
        std::cout << std::endl;
        
        // Recursively process next example
        self(self, index + 1);
    };
    
    processExamples(processExamples, 0);
    
    // Word cloud visualization
    std::cout << "Generating word clouds..." << std::endl;
    analyzer.generateWordCloud(dataset, 4); // positive sentiment
    analyzer.generateWordCloud(dataset, 0); // negative sentiment
    
    std::cout << "\n==== Demo Complete ====" << std::endl;
}

/**
 * Main function for the example application.
 * 
 * @return Exit status code
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

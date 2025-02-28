#include "sentiment_analysis.h"
#include <iostream>
#include <cassert>
#include <string>
#include <vector>

// Simple test framework
#define TEST(name) void name()
#define RUN_TEST(name) std::cout << "Running " << #name << "... "; name(); std::cout << "PASSED" << std::endl

// Test TextPreprocessor functionality
TEST(testTextPreprocessor) {
    nlp::TextPreprocessor preprocessor;
    
    // Test with a typical tweet
    std::string input = "@user I LOVED the movie!!! It's amazing & worth $12.99 :) #mustwatch";
    std::string processed = preprocessor.preprocess(input);
    
    // Check basic expectations
    assert(!processed.empty());
    
    // Should be lowercase
    for (char c : processed) {
        if (c >= 'A' && c <= 'Z') {
            assert(false && "Text should be lowercase");
        }
    }
    
    // Should not contain punctuation
    assert(processed.find('!') == std::string::npos);
    assert(processed.find('@') == std::string::npos);
    assert(processed.find('#') == std::string::npos);
    assert(processed.find('$') == std::string::npos);
    
    // Should not contain numbers
    assert(processed.find('1') == std::string::npos);
    assert(processed.find('2') == std::string::npos);
    assert(processed.find('9') == std::string::npos);
    
    // Test with empty string
    assert(preprocessor.preprocess("").empty());
    
    // Test with only stopwords
    std::string allStopwords = "the and a of";
    std::string processedStopwords = preprocessor.preprocess(allStopwords);
    assert(processedStopwords.empty());
}

// Test TfidfVectorizer functionality
TEST(testTfidfVectorizer) {
    nlp::TfidfVectorizer vectorizer;
    
    // Create a small corpus
    std::vector<std::string> corpus = {
        "this is the first document",
        "this document is the second document",
        "and this is the third one",
        "is this the first document"
    };
    
    // Fit and transform
    std::vector<std::vector<double>> features = vectorizer.fitTransform(corpus);
    
    // Check dimensions
    assert(features.size() == corpus.size());
    
    // All feature vectors should be the same length
    size_t featureSize = features[0].size();
    for (const auto& feature : features) {
        assert(feature.size() == featureSize);
    }
    
    // Check that transformation of new documents works
    std::vector<std::string> newDocs = {"this is a new document"};
    std::vector<std::vector<double>> newFeatures = vectorizer.transform(newDocs);
    
    assert(newFeatures.size() == newDocs.size());
    assert(newFeatures[0].size() == featureSize);
}

// Test SGDClassifier functionality
TEST(testSGDClassifier) {
    nlp::SGDClassifier classifier;
    
    // Create a simple binary classification problem
    std::vector<std::vector<double>> X = {
        {1.0, 2.0}, {2.0, 3.0}, {3.0, 4.0}, {4.0, 5.0},  // Class 0
        {-1.0, -2.0}, {-2.0, -3.0}, {-3.0, -4.0}, {-4.0, -5.0}  // Class 4
    };
    
    std::vector<int> y = {0, 0, 0, 0, 4, 4, 4, 4};
    
    // Train the classifier
    classifier.fit(X, y);
    
    // Predict
    std::vector<int> predictions = classifier.predict(X);
    assert(predictions.size() == X.size());
    
    // Calculate accuracy
    double accuracy = classifier.score(X, y);
    assert(accuracy >= 0.0 && accuracy <= 1.0);
    
    // Test decision function
    std::vector<double> decisions = classifier.decisionFunction(X);
    assert(decisions.size() == X.size());
}

// Test ModelEvaluator functionality
TEST(testModelEvaluator) {
    // Create simple test data
    std::vector<int> yTrue = {0, 0, 0, 0, 4, 4, 4, 4};
    std::vector<int> yPred = {0, 0, 4, 0, 4, 4, 0, 4};
    
    // Test confusion matrix
    std::vector<std::vector<int>> confusionMatrix = nlp::ModelEvaluator::confusionMatrix(yTrue, yPred);
    assert(confusionMatrix.size() == 2);  // 2 classes
    assert(confusionMatrix[0].size() == 2);
    
    // True Positives for class 0
    assert(confusionMatrix[0][0] == 3);
    
    // False Negatives for class 0
    assert(confusionMatrix[0][1] == 1);
    
    // False Positives for class 0
    assert(confusionMatrix[1][0] == 1);
    
    // True Positives for class 4
    assert(confusionMatrix[1][1] == 3);
    
    // Test classification report
    std::string report = nlp::ModelEvaluator::classificationReport(
        yTrue, yPred, {"Negative", "Positive"}
    );
    
    assert(!report.empty());
}

// Basic test for SentimentAnalyzer
TEST(testSentimentAnalyzer) {
    nlp::SentimentAnalyzer analyzer;
    
    // Just a basic initialization check - no meaningful assertion needed here
    // The existence of the analyzer object is already verified by reaching this point
    
    // Can't do much more without actual data, but we can test that it doesn't crash
    try {
        // These should fail gracefully without crashing
        bool loadResult = analyzer.loadData("nonexistent_file.csv");
        assert(loadResult == false);
        
        bool preprocessResult = analyzer.preprocessData();
        assert(preprocessResult == false);
    } catch (const std::exception& e) {
        std::cerr << "Unexpected exception: " << e.what() << std::endl;
        assert(false);
    }
}

int main() {
    std::cout << "======== Running NLP Sentiment Analysis Tests ========" << std::endl;
    
    RUN_TEST(testTextPreprocessor);
    RUN_TEST(testTfidfVectorizer);
    RUN_TEST(testSGDClassifier);
    RUN_TEST(testModelEvaluator);
    RUN_TEST(testSentimentAnalyzer);
    
    std::cout << "All tests passed!" << std::endl;
    return 0;
}

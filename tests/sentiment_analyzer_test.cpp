/**
 * @file sentiment_analyzer_test.cpp
 * @brief Tests for the sentiment analyzer component
 */

#include "sentiment_analysis.h"
#include "classifier_model.h"
#include <iostream>
#include <cassert>
#include <string>
#include <cmath>

// Simple test framework
#define TEST(name) void name()
#define RUN_TEST(name) std::cout << "Running " << #name << "... "; name(); std::cout << "PASSED" << std::endl

/**
 * Test basic SentimentAnalyzer functionality.
 * 
 * This function tests the main sentiment analysis component,
 * focusing on initialization and error handling. It demonstrates
 * a functional approach to boundary case testing.
 */
TEST(testSentimentAnalyzer) {
    nlp::SentimentAnalyzer analyzer;
    
    // Test prediction with a minimal model
    try {
        // Create a mock model for testing
        class MockModel : public nlp::ClassifierModel {
        public:
            void fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y) override {
                // Do nothing for mock
            }
            
            std::vector<int> predict(const std::vector<std::vector<double>>& X) const override {
                // Always predict positive for mock
                return std::vector<int>(X.size(), 4);
            }
            
            std::vector<double> decisionFunction(const std::vector<std::vector<double>>& X) const override {
                // Return positive scores for mock
                return std::vector<double>(X.size(), 1.5);
            }
            
            double score(const std::vector<std::vector<double>>& X, const std::vector<int>& y) const override {
                return 0.75; // Mock accuracy
            }
            
            std::vector<double> predict_proba(const std::vector<std::vector<double>>& X) const override {
                // Return probabilities > 0.5 for mock
                return std::vector<double>(X.size(), 0.8);
            }
        };
        
        // Create a mock vectorizer for testing
        nlp::TfidfVectorizer mockVectorizer;
        
        // Build vocabulary with some sample texts
        std::vector<std::string> sampleTexts = {
            "this is a positive review",
            "this is a negative review"
        };
        mockVectorizer.fitTransform(sampleTexts);
        
        // Create mock model
        MockModel mockModel;
        
        // Test sentiment prediction
        nlp::SentimentResult result = analyzer.predictSentiment("This is a test", mockModel, mockVectorizer);
        
        // Verify result properties
        assert(result.text == "This is a test");
        assert(!result.cleanText.empty());
        assert(result.sentiment == "Positive"); // Our mock always predicts positive
        assert(result.label == 4); // From our mock prediction
        assert(result.rawScore == 1.5); // From our mock decision function
        assert(result.confidence > 0.5); // Should be > 0.5 for positive prediction
        assert(!result.explanation.empty());
        
        // Test serialization and deserialization
        auto resultMap = result.toMap();
        nlp::SentimentResult reconstructed = nlp::SentimentResult::fromMap(resultMap);
        
        assert(reconstructed.text == result.text);
        assert(reconstructed.sentiment == result.sentiment);
        assert(std::abs(reconstructed.rawScore - result.rawScore) < 1e-6);
    } catch (const std::exception& e) {
        std::cerr << "Unexpected exception: " << e.what() << std::endl;
        assert(false && "No exception should be thrown during mock testing");
    }
    
    // Test with invalid data paths
    auto invalidDataset = analyzer.loadData("nonexistent_file.csv");
    assert(!invalidDataset.has_value() && "Should return empty optional for invalid file");
}

int main() {
    std::cout << "==== SentimentAnalyzer Tests ====" << std::endl;
    testSentimentAnalyzer();
    std::cout << "All SentimentAnalyzer tests passed!" << std::endl;
    return 0;
}

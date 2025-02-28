/**
 * @file nlp_tests.cpp
 * @brief Comprehensive test suite for the NLP sentiment analysis library
 * 
 * This file contains test functions that verify the correctness of
 * various components in the sentiment analysis library. It demonstrates
 * a functional approach to testing using recursion instead of loops.
 */

#include "sentiment_analysis.h"
#include <iostream>
#include <cassert>
#include <string>
#include <vector>
#include <functional>

// Simple test framework
#define TEST(name) void name()
#define RUN_TEST(name) std::cout << "Running " << #name << "... "; name(); std::cout << "PASSED" << std::endl

/**
 * Test TextPreprocessor functionality.
 * 
 * This function tests the text preprocessing component, ensuring
 * that it properly cleans and normalizes text data according to
 * expectations. It uses a functional approach with recursive checks.
 */
TEST(testTextPreprocessor) {
    nlp::TextPreprocessor preprocessor;
    
    // Test with a typical tweet
    std::string input = "@user I LOVED the movie!!! It's amazing & worth $12.99 :) #mustwatch";
    std::string processed = preprocessor.preprocess(input);
    
    // Check basic expectations
    assert(!processed.empty());
    
    // Recursively check if all characters are lowercase
    const auto checkAllLowercase = [](auto& self, const std::string& text, size_t index) -> bool {
        // Base case: all characters checked
        if (index >= text.size()) {
            return true;
        }
        
        // Check if this character is uppercase
        if (std::isupper(static_cast<unsigned char>(text[index]))) {
            return false;
        }
        
        // Recursively check next character
        return self(self, text, index + 1);
    };
    
    assert(checkAllLowercase(checkAllLowercase, processed, 0) && "Text should be lowercase");
    
    // Recursively check for specific characters
    const auto checkForChar = [](auto& self, const std::string& text, char target, size_t index) -> bool {
        // Base case: character not found in remaining text
        if (index >= text.size()) {
            return false;
        }
        
        // Check if this is the target character
        if (text[index] == target) {
            return true;
        }
        
        // Recursively check next character
        return self(self, text, target, index + 1);
    };
    
    // Should not contain punctuation
    assert(!checkForChar(checkForChar, processed, '!', 0));
    assert(!checkForChar(checkForChar, processed, '@', 0));
    assert(!checkForChar(checkForChar, processed, '#', 0));
    assert(!checkForChar(checkForChar, processed, '$', 0));
    
    // Should not contain numbers
    assert(!checkForChar(checkForChar, processed, '1', 0));
    assert(!checkForChar(checkForChar, processed, '2', 0));
    assert(!checkForChar(checkForChar, processed, '9', 0));
    
    // Test with empty string
    assert(preprocessor.preprocess("").empty());
    
    // Test with only stopwords
    std::string allStopwords = "the and a of";
    std::string processedStopwords = preprocessor.preprocess(allStopwords);
    assert(processedStopwords.empty());
    
    // Test individual preprocessing functions
    auto preprocessingFuncs = preprocessor.getPreprocessingFunctions();
    
    // Test lowercase function
    std::string upperText = "ALL UPPERCASE TEXT";
    std::string lowerText = preprocessingFuncs["lowercase"](upperText);
    assert(checkAllLowercase(checkAllLowercase, lowerText, 0) && "Lowercase function failed");
    
    // Test remove_punctuation function
    std::string punctText = "Text, with. punctuation!";
    std::string noPunctText = preprocessingFuncs["remove_punctuation"](punctText);
    assert(!checkForChar(checkForChar, noPunctText, ',', 0) && "Punctuation removal failed");
    assert(!checkForChar(checkForChar, noPunctText, '.', 0) && "Punctuation removal failed");
    assert(!checkForChar(checkForChar, noPunctText, '!', 0) && "Punctuation removal failed");
}

/**
 * Test TfidfVectorizer functionality.
 * 
 * This function tests the feature extraction component, ensuring
 * that it correctly transforms text into numerical feature vectors.
 * It demonstrates a functional approach to testing vector properties.
 */
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
    
    // Check that all feature vectors have the same length using recursion
    const auto checkFeatureSize = [](auto& self, const auto& features, 
                                size_t index, size_t expectedSize) -> bool {
        // Base case: all features checked
        if (index >= features.size()) {
            return true;
        }
        
        // Check this feature vector's size
        if (features[index].size() != expectedSize) {
            return false;
        }
        
        // Recursively check next feature vector
        return self(self, features, index + 1, expectedSize);
    };
    
    size_t featureSize = features[0].size();
    assert(checkFeatureSize(checkFeatureSize, features, 0, featureSize));
    
    // Recursively check for non-zero values (at least one feature should be non-zero)
    const auto hasNonZeroValue = [](auto& self, const auto& feature, size_t index) -> bool {
        // Base case: no non-zero value found
        if (index >= feature.size()) {
            return false;
        }
        
        // Check if this feature is non-zero
        if (std::abs(feature[index]) > 1e-10) {
            return true;
        }
        
        // Recursively check next feature
        return self(self, feature, index + 1);
    };
    
    // Check that each document has at least one non-zero feature
    const auto checkAllFeaturesHaveValues = [&](auto& self, size_t index) -> bool {
        // Base case: all features checked
        if (index >= features.size()) {
            return true;
        }
        
        // Check this feature vector
        if (!hasNonZeroValue(hasNonZeroValue, features[index], 0)) {
            return false;
        }
        
        // Recursively check next feature vector
        return self(self, index + 1);
    };
    
    assert(checkAllFeaturesHaveValues(checkAllFeaturesHaveValues, 0) && "Features should have non-zero values");
    
    // Check that transformation of new documents works
    std::vector<std::string> newDocs = {"this is a new document"};
    std::vector<std::vector<double>> newFeatures = vectorizer.transform(newDocs);
    
    assert(newFeatures.size() == newDocs.size());
    assert(newFeatures[0].size() == featureSize);
}

/**
 * Test SGDClassifier functionality.
 * 
 * This function tests the machine learning component, ensuring
 * that it correctly trains on data and makes predictions.
 * It uses a functional approach to create and verify the model.
 */
TEST(testSGDClassifier) {
    nlp::SGDClassifier classifier;
    
    // Create a simple binary classification problem with clean separation
    std::vector<std::vector<double>> X = {
        {1.0, 2.0}, {2.0, 3.0}, {3.0, 4.0}, {4.0, 5.0},  // Class 0
        {-1.0, -2.0}, {-2.0, -3.0}, {-3.0, -4.0}, {-4.0, -5.0}  // Class 4
    };
    
    std::vector<int> y = {0, 0, 0, 0, 4, 4, 4, 4};
    
    // Train the classifier
    classifier.fit(X, y);
    
    // Predict using recursion
    const auto makePredictions = [&classifier](auto& self, const auto& X, 
                                         size_t index, std::vector<int>& predictions) -> void {
        // Base case: all samples predicted
        if (index >= X.size()) {
            return;
        }
        
        // Make prediction for this sample (we'll use the full predict method for simplicity)
        predictions.push_back(classifier.predict({X[index]})[0]);
        
        // Recursively predict next sample
        self(self, X, index + 1, predictions);
    };
    
    std::vector<int> predictions;
    makePredictions(makePredictions, X, 0, predictions);
    
    assert(predictions.size() == X.size());
    
    // Calculate accuracy manually using recursion
    const auto calculateAccuracy = [](auto& self, const auto& actual, const auto& predicted, 
                                 size_t index, int correct) -> double {
        // Base case: all predictions checked
        if (index >= actual.size()) {
            return static_cast<double>(correct) / actual.size();
        }
        
        // Check if this prediction is correct
        int newCorrect = correct + (actual[index] == predicted[index] ? 1 : 0);
        
        // Recursively check next prediction
        return self(self, actual, predicted, index + 1, newCorrect);
    };
    
    double accuracy = calculateAccuracy(calculateAccuracy, y, predictions, 0, 0);
    
    // The problem is cleanly separable, so we should get high accuracy
    assert(accuracy >= 0.7 && "Classifier should achieve reasonable accuracy on separable data");
    
    // Test classifier's own score method
    double modelScore = classifier.score(X, y);
    assert(modelScore >= 0.0 && modelScore <= 1.0);
    
    // Test decision function
    std::vector<double> decisions;
    
    const auto makeDecisions = [&classifier](auto& self, const auto& X, 
                                      size_t index, std::vector<double>& decisions) -> void {
        // Base case: all samples processed
        if (index >= X.size()) {
            return;
        }
        
        // Get decision value for this sample
        auto sampleDecisions = classifier.decisionFunction({X[index]});
        decisions.push_back(sampleDecisions[0]);
        
        // Recursively process next sample
        self(self, X, index + 1, decisions);
    };
    
    makeDecisions(makeDecisions, X, 0, decisions);
    
    assert(decisions.size() == X.size());
    
    // Verifies that positive examples have positive decision values
    // and negative examples have negative decision values
    const auto checkDecisionSigns = [](auto& self, const auto& y, const auto& decisions, 
                                 size_t index) -> bool {
        // Base case: all decisions checked
        if (index >= y.size()) {
            return true;
        }
        
        // Check if the decision value sign matches the class
        // Class 0 (negative) should have negative decision value
        // Class 4 (positive) should have positive decision value
        bool correctSign = ((y[index] == 0 && decisions[index] < 0) || 
                          (y[index] == 4 && decisions[index] > 0));
        
        if (!correctSign) {
            return false;
        }
        
        // Recursively check next decision
        return self(self, y, decisions, index + 1);
    };
    
    assert(checkDecisionSigns(checkDecisionSigns, y, decisions, 0) && 
           "Decision values should match expected class signs");
}

/**
 * Test ModelEvaluator functionality.
 * 
 * This function tests the evaluation component, ensuring that it
 * correctly calculates performance metrics for classification results.
 * It demonstrates a functional approach to metric verification.
 */
TEST(testModelEvaluator) {
    // Create simple test data
    std::vector<int> yTrue = {0, 0, 0, 0, 4, 4, 4, 4};
    std::vector<int> yPred = {0, 0, 4, 0, 4, 4, 0, 4};
    
    // Test confusion matrix
    std::vector<std::vector<int>> confusionMatrix = nlp::ModelEvaluator::confusionMatrix(yTrue, yPred);
    assert(confusionMatrix.size() == 2);  // 2 classes
    assert(confusionMatrix[0].size() == 2);
    
    // Verify confusion matrix values
    // For class 0: 3 true positives, 1 false negative
    assert(confusionMatrix[0][0] == 3 && "True positives for class 0 should be 3");
    assert(confusionMatrix[0][1] == 1 && "False negatives for class 0 should be 1");
    
    // For class 4: 3 true positives, 1 false positive
    assert(confusionMatrix[1][0] == 1 && "False positives for class 4 should be 1");
    assert(confusionMatrix[1][1] == 3 && "True positives for class 4 should be 3");
    
    // Test classification report
    std::string report = nlp::ModelEvaluator::classificationReport(
        yTrue, yPred, {"Negative", "Positive"}
    );
    
    assert(!report.empty());
    
    // Test metrics calculation
    auto metrics = nlp::ModelEvaluator::calculateMetrics(yTrue, yPred);
    
    assert(metrics.contains("accuracy"));
    assert(metrics["accuracy"] >= 0.0 && metrics["accuracy"] <= 1.0);
    
    assert(metrics.contains("macro_precision"));
    assert(metrics["macro_precision"] >= 0.0 && metrics["macro_precision"] <= 1.0);
    
    assert(metrics.contains("macro_recall"));
    assert(metrics["macro_recall"] >= 0.0 && metrics["macro_recall"] <= 1.0);
    
    assert(metrics.contains("macro_f1"));
    assert(metrics["macro_f1"] >= 0.0 && metrics["macro_f1"] <= 1.0);
    
    // Given our predictions, we expect 75% accuracy (6 correct out of 8)
    assert(std::abs(metrics["accuracy"] - 0.75) < 1e-6 && "Accuracy should be 0.75");
}

/**
 * Test AsciiWordCloud functionality.
 * 
 * This function tests the visualization component, ensuring that it
 * correctly generates word clouds from text data. It uses a functional
 * approach to verify cloud properties.
 */
TEST(testAsciiWordCloud) {
    // Create test texts
    std::vector<std::string> texts = {
        "great amazing wonderful excellent",
        "great wonderful excellent product",
        "amazing product great review",
        "excellent service great experience"
    };
    
    // Generate word cloud
    std::string cloud = nlp::AsciiWordCloud::generateWordCloud(texts);
    
    // Check that the cloud contains all our frequent words
    const auto containsWord = [](auto& self, const std::string& text, 
                            const std::string& word, size_t index) -> bool {
        // Base case: word not found in remaining text
        if (index > text.size() - word.size()) {
            return false;
        }
        
        // Check if the word is at this position
        if (text.substr(index, word.size()) == word) {
            return true;
        }
        
        // Recursively check next position
        return self(self, text, word, index + 1);
    };
    
    assert(containsWord(containsWord, cloud, "great", 0) && "Cloud should contain 'great'");
    assert(containsWord(containsWord, cloud, "amazing", 0) && "Cloud should contain 'amazing'");
    assert(containsWord(containsWord, cloud, "wonderful", 0) && "Cloud should contain 'wonderful'");
    assert(containsWord(containsWord, cloud, "excellent", 0) && "Cloud should contain 'excellent'");
    
    // Test custom cloud configuration
    nlp::AsciiWordCloud::CloudConfig config;
    config.maxWords = 3;
    config.width = 40;
    config.height = 5;
    config.useColor = false;
    
    std::string customCloud = nlp::AsciiWordCloud::generateCustomCloud(texts, config);
    
    // The custom cloud should be smaller and contain only top words
    assert(customCloud.size() < cloud.size());
    
    // Test with empty texts
    std::vector<std::string> emptyTexts;
    std::string emptyCloud = nlp::AsciiWordCloud::generateWordCloud(emptyTexts);
    
    assert(containsWord(containsWord, emptyCloud, "No words found", 0) && 
           "Empty cloud should contain error message");
}

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
        assert(result.rawScore == 1.5); // From our mock decision function
        assert(result.probability > 0.5); // Should be > 0.5 for positive prediction
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

/**
 * Main function runs all tests.
 * 
 * This function orchestrates the test suite, running each test
 * and reporting the overall results.
 * 
 * @return Exit status code
 */
int main() {
    std::cout << "======== Running NLP Sentiment Analysis Tests ========" << std::endl;
    
    // Run all tests using a recursive approach
    const auto runAllTests = [](auto& self, size_t testIndex, 
                            const std::vector<std::pair<std::string, std::function<void()>>>& tests) -> void {
        // Base case: all tests run
        if (testIndex >= tests.size()) {
            return;
        }
        
        // Run this test
        const auto& [name, testFunc] = tests[testIndex];
        std::cout << "Running " << name << "... ";
        testFunc();
        std::cout << "PASSED" << std::endl;
        
        // Recursively run next test
        self(self, testIndex + 1, tests);
    };
    
    // Define all tests
    std::vector<std::pair<std::string, std::function<void()>>> tests = {
        {"testTextPreprocessor", testTextPreprocessor},
        {"testTfidfVectorizer", testTfidfVectorizer},
        {"testSGDClassifier", testSGDClassifier},
        {"testModelEvaluator", testModelEvaluator},
        {"testAsciiWordCloud", testAsciiWordCloud},
        {"testSentimentAnalyzer", testSentimentAnalyzer}
    };
    
    // Run all tests
    runAllTests(runAllTests, 0, tests);
    
    std::cout << "All tests passed!" << std::endl;
    return 0;
}

/**
 * @file sgd_classifier_test.cpp
 * @brief Tests for the SGD classifier component
 */

#include "sgd_classifier.h"
#include <iostream>
#include <cassert>
#include <vector>

// Simple test framework
#define TEST(name) void name()
#define RUN_TEST(name) std::cout << "Running " << #name << "... "; name(); std::cout << "PASSED" << std::endl

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
        
        // Make prediction for this sample
        std::vector<std::vector<double>> sample = {X[index]};
        predictions.push_back(classifier.predict(sample)[0]);
        
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
        std::vector<std::vector<double>> sample = {X[index]};
        auto sampleDecisions = classifier.decisionFunction(sample);
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

int main() {
    std::cout << "==== SGDClassifier Tests ====" << std::endl;
    testSGDClassifier();
    std::cout << "All SGDClassifier tests passed!" << std::endl;
    return 0;
}

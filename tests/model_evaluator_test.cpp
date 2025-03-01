/**
 * @file model_evaluator_test.cpp
 * @brief Tests for the model evaluator component
 */

#include "model_evaluator.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

// Simple test framework
#define TEST(name) void name()
#define RUN_TEST(name) std::cout << "Running " << #name << "... "; name(); std::cout << "PASSED" << std::endl

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

int main() {
    std::cout << "==== ModelEvaluator Tests ====" << std::endl;
    testModelEvaluator();
    std::cout << "All ModelEvaluator tests passed!" << std::endl;
    return 0;
}

/**
 * @file model_evaluator.cpp
 * @brief Implements evaluation metrics for classification models.
 *
 * This file provides implementations for computing performance metrics
 * such as confusion matrix, precision, recall, and F1-score for binary
 * classification models. These metrics help assess model quality and
 * guide improvement efforts.
 */

#include "model_evaluator.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cmath>

namespace nlp {

/**
 * @brief Computes the confusion matrix for binary classification.
 * 
 * The confusion matrix shows the counts of true positives, false positives,
 * true negatives, and false negatives in a 2x2 matrix format.
 * 
 * @param yTrue Ground truth labels (0 for negative, 4 for positive).
 * @param yPred Predicted labels (0 for negative, 4 for positive).
 * @return 2D confusion matrix as [[TN, FP], [FN, TP]].
 */
std::vector<std::vector<int>> ModelEvaluator::confusionMatrix(
    const std::vector<int>& yTrue,
    const std::vector<int>& yPred) {
    
    // Initialize 2x2 confusion matrix
    std::vector<std::vector<int>> matrix(2, std::vector<int>(2, 0));
    
    // Count each prediction type
    for (size_t i = 0; i < yTrue.size() && i < yPred.size(); ++i) {
        // Convert labels from {0,4} to {0,1} for matrix indexing
        int actual = yTrue[i] == 4 ? 1 : 0;
        int predicted = yPred[i] == 4 ? 1 : 0;
        
        // Increment the appropriate cell in the confusion matrix
        matrix[actual][predicted]++;
    }
    
    return matrix;
}

/**
 * @brief Generates a classification report with precision, recall, and F1-score.
 * 
 * This method creates a formatted string containing key performance metrics
 * for each class and the overall model, suitable for display or logging.
 * 
 * @param yTrue Ground truth labels.
 * @param yPred Predicted labels.
 * @param classNames Optional names for the classes.
 * @return Formatted classification report as a string.
 */
std::string ModelEvaluator::classificationReport(
    const std::vector<int>& yTrue,
    const std::vector<int>& yPred,
    const std::vector<std::string>& classNames) {
    
    // Compute confusion matrix
    auto matrix = confusionMatrix(yTrue, yPred);
    
    // Get matrix values
    int tn = matrix[0][0];  // True Negatives
    int fp = matrix[0][1];  // False Positives
    int fn = matrix[1][0];  // False Negatives
    int tp = matrix[1][1];  // True Positives
    
    // Calculate metrics
    double precisionNeg = precision(tn, fn);
    double precisionPos = precision(tp, fp);
    double recallNeg = recall(tn, fp);
    double recallPos = recall(tp, fn);
    double f1Neg = f1Score(precisionNeg, recallNeg);
    double f1Pos = f1Score(precisionPos, recallPos);
    
    // Calculate macro-averaged metrics
    double macroPrec = (precisionNeg + precisionPos) / 2.0;
    double macroRec = (recallNeg + recallPos) / 2.0;
    double macroF1 = (f1Neg + f1Pos) / 2.0;
    
    // Calculate accuracy
    int totalSamples = tn + fp + fn + tp;
    double accuracy = totalSamples > 0 ? 
                     static_cast<double>(tn + tp) / totalSamples : 0.0;
    
    // Format the report
    std::ostringstream report;
    report << std::fixed << std::setprecision(2);
    
    // Header
    report << "Classification Report:\n";
    report << std::setw(20) << "precision" << std::setw(10) << "recall"
           << std::setw(10) << "f1-score" << std::setw(10) << "support" << "\n";
    report << std::string(50, '-') << "\n";
    
    // Class: Negative
    report << std::setw(20) << classNames[0] << std::setw(10) << (precisionNeg * 100)
           << std::setw(10) << (recallNeg * 100) << std::setw(10) << (f1Neg * 100)
           << std::setw(10) << (tn + fp) << "\n";
    
    // Class: Positive
    report << std::setw(20) << classNames[1] << std::setw(10) << (precisionPos * 100)
           << std::setw(10) << (recallPos * 100) << std::setw(10) << (f1Pos * 100)
           << std::setw(10) << (tp + fn) << "\n";
    
    report << std::string(50, '-') << "\n";
    
    // Macro average
    report << std::setw(20) << "macro avg" << std::setw(10) << (macroPrec * 100)
           << std::setw(10) << (macroRec * 100) << std::setw(10) << (macroF1 * 100)
           << std::setw(10) << totalSamples << "\n";
    
    // Accuracy
    report << std::setw(20) << "accuracy" << std::setw(30) << (accuracy * 100)
           << std::setw(10) << totalSamples << "\n";
    
    return report.str();
}

/**
 * @brief Computes precision, recall, F1-score, and accuracy metrics.
 * 
 * This method calculates all standard classification metrics from the
 * predicted and true labels, returning them in a convenient map.
 * 
 * @param yTrue Ground truth labels.
 * @param yPred Predicted labels.
 * @return Map of metric names to values.
 */
std::unordered_map<std::string, double> ModelEvaluator::calculateMetrics(
    const std::vector<int>& yTrue,
    const std::vector<int>& yPred) {
    
    // Initialize counts
    int tp = 0, fp = 0, fn = 0, tn = 0;
    
    // Count each prediction type
    for (size_t i = 0; i < yTrue.size() && i < yPred.size(); ++i) {
        if (yTrue[i] == 4 && yPred[i] == 4) tp++;
        else if (yTrue[i] == 0 && yPred[i] == 4) fp++;
        else if (yTrue[i] == 4 && yPred[i] == 0) fn++;
        else if (yTrue[i] == 0 && yPred[i] == 0) tn++;
    }
    
    // Calculate metrics
    double prec = precision(tp, fp);
    double rec = recall(tp, fn);
    double f1 = f1Score(prec, rec);
    double accuracy = static_cast<double>(tp + tn) / (tp + fp + fn + tn);
    
    // For negative class
    double precNeg = precision(tn, fn);
    double recNeg = recall(tn, fp);
    double f1Neg = f1Score(precNeg, recNeg);
    
    // Macro averages
    double macroPrec = (prec + precNeg) / 2.0;
    double macroRec = (rec + recNeg) / 2.0;
    double macroF1 = (f1 + f1Neg) / 2.0;
    
    // Return all metrics in a map
    return {
        {"precision", prec},
        {"recall", rec},
        {"f1_score", f1},
        {"accuracy", accuracy},
        {"macro_precision", macroPrec},
        {"macro_recall", macroRec},
        {"macro_f1", macroF1}
    };
}

/**
 * @brief Calculates precision score.
 * 
 * Precision is the ratio of correctly predicted positive observations
 * to the total predicted positives: TP / (TP + FP).
 * 
 * @param tp True positives.
 * @param fp False positives.
 * @return Precision score or 0 if undefined.
 */
double ModelEvaluator::precision(int tp, int fp) {
    return (tp + fp > 0) ? static_cast<double>(tp) / (tp + fp) : 0.0;
}

/**
 * @brief Calculates recall score.
 * 
 * Recall is the ratio of correctly predicted positive observations
 * to all actual positives: TP / (TP + FN).
 * 
 * @param tp True positives.
 * @param fn False negatives.
 * @return Recall score or 0 if undefined.
 */
double ModelEvaluator::recall(int tp, int fn) {
    return (tp + fn > 0) ? static_cast<double>(tp) / (tp + fn) : 0.0;
}

/**
 * @brief Calculates F1 score.
 * 
 * F1 Score is the harmonic mean of Precision and Recall,
 * providing a balance between them: 2 * (Precision * Recall) / (Precision + Recall).
 * 
 * @param precision Precision value.
 * @param recall Recall value.
 * @return F1 score or 0 if undefined.
 */
double ModelEvaluator::f1Score(double precision, double recall) {
    return (precision + recall > 0) ? 
           2.0 * (precision * recall) / (precision + recall) : 0.0;
}

}  // namespace nlp

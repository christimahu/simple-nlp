/**
 * @file model_evaluator.h
 * @brief Provides functions to evaluate classifier performance.
 *
 * This module includes metrics like confusion matrix, precision, recall, 
 * F1-score, and classification report generation for evaluating the 
 * performance of sentiment classification models.
 */

#pragma once

#include <vector>
#include <unordered_map>
#include <string>

namespace nlp {

/**
 * @class ModelEvaluator
 * @brief Evaluates the performance of classification models.
 * 
 * This class provides static methods to compute various evaluation
 * metrics for classification tasks. It focuses on binary classification
 * metrics relevant to sentiment analysis.
 */
class ModelEvaluator {
public:
    /**
     * @brief Computes the confusion matrix.
     * 
     * The confusion matrix shows the counts of true positives (TP),
     * false positives (FP), true negatives (TN), and false negatives (FN).
     * 
     * @param yTrue Ground truth labels.
     * @param yPred Predicted labels.
     * @return 2D confusion matrix as [[TN, FP], [FN, TP]].
     */
    static std::vector<std::vector<int>> confusionMatrix(
        const std::vector<int>& yTrue,
        const std::vector<int>& yPred);

    /**
     * @brief Generates a classification report with precision, recall, and F1-score.
     * 
     * This report provides a textual summary of the classification performance,
     * suitable for displaying to users or logging.
     * 
     * @param yTrue Ground truth labels.
     * @param yPred Predicted labels.
     * @param classNames Optional names for the classes (defaults to "Negative"/"Positive").
     * @return Formatted classification report as a string.
     */
    static std::string classificationReport(
        const std::vector<int>& yTrue,
        const std::vector<int>& yPred,
        const std::vector<std::string>& classNames = {"Negative", "Positive"});

    /**
     * @brief Computes precision, recall, F1-score, and accuracy.
     * 
     * This method calculates all standard classification metrics in one call,
     * returning them in a map for convenient access.
     * 
     * @param yTrue Ground truth labels.
     * @param yPred Predicted labels.
     * @return A map containing precision, recall, F1-score, and accuracy.
     */
    static std::unordered_map<std::string, double> calculateMetrics(
        const std::vector<int>& yTrue,
        const std::vector<int>& yPred);

private:
    /**
     * @brief Calculates the precision score.
     * @param tp True positives.
     * @param fp False positives.
     * @return Precision score or 0 if undefined.
     */
    static double precision(int tp, int fp);

    /**
     * @brief Calculates the recall score.
     * @param tp True positives.
     * @param fn False negatives.
     * @return Recall score or 0 if undefined.
     */
    static double recall(int tp, int fn);

    /**
     * @brief Calculates the F1 score.
     * @param precision Precision value.
     * @param recall Recall value.
     * @return F1 score or 0 if undefined.
     */
    static double f1Score(double precision, double recall);
};

}  // namespace nlp

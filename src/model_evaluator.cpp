/**
 * @file model_evaluator.cpp
 * @brief Implementation of the ModelEvaluator class
 * 
 * This file contains the implementation of model evaluation methods,
 * including confusion matrices, classification reports, and performance metrics.
 * These tools help assess the quality of sentiment classification.
 */

#include "sentiment_analysis.h"
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <set>
#include <span>
#include <unordered_map>
#include <ranges>
#include <numeric>

namespace nlp {

/**
 * Generates a confusion matrix from true and predicted labels.
 * 
 * A confusion matrix shows the counts of true positives, false positives,
 * true negatives, and false negatives, which are essential for
 * evaluating classification performance.
 * 
 * @param yTrue Vector of true labels
 * @param yPred Vector of predicted labels
 * @return Confusion matrix as a 2D vector
 */
std::vector<std::vector<int>> ModelEvaluator::confusionMatrix(
    std::span<const int> yTrue, std::span<const int> yPred) {
    
    // Check for empty or mismatched input
    if (yTrue.empty() || yPred.empty() || yTrue.size() != yPred.size()) {
        return {{0}};
    }
    
    // Identify unique classes through recursion
    const auto findUniqueClasses = [](auto& self, std::span<const int> labels, 
                                   size_t index, std::set<int>& classes) -> void {
        // Base case: all labels processed
        if (index >= labels.size()) {
            return;
        }
        
        // Add this label to the set and recurse
        classes.insert(labels[index]);
        self(self, labels, index + 1, classes);
    };
    
    std::set<int> classes;
    findUniqueClasses(findUniqueClasses, yTrue, 0, classes);
    findUniqueClasses(findUniqueClasses, yPred, 0, classes);
    
    // Map class values to matrix indices
    std::unordered_map<int, size_t> classIndices;
    size_t idx = 0;
    for (int classVal : classes) {
        classIndices[classVal] = idx++;
    }
    
    // Initialize confusion matrix with zeros
    std::vector<std::vector<int>> matrix(classes.size(), std::vector<int>(classes.size(), 0));
    
    // Fill the confusion matrix through recursion
    const auto fillMatrix = [&](auto& self, size_t index) -> void {
        // Base case: all predictions processed
        if (index >= yTrue.size()) {
            return;
        }
        
        // Increment the appropriate cell in the matrix
        size_t trueIdx = classIndices[yTrue[index]];
        size_t predIdx = classIndices[yPred[index]];
        matrix[trueIdx][predIdx]++;
        
        // Recursively process the next prediction
        self(self, index + 1);
    };
    
    fillMatrix(fillMatrix, 0);
    
    return matrix;
}

/**
 * Generates a classification report with precision, recall, and F1-score.
 * 
 * This report provides a detailed breakdown of classifier performance
 * for each class and overall, making it easier to understand strengths
 * and weaknesses.
 * 
 * @param yTrue Vector of true labels
 * @param yPred Vector of predicted labels
 * @param targetNames Names of the classes
 * @return Classification report as a formatted string
 */
std::string ModelEvaluator::classificationReport(
    std::span<const int> yTrue, std::span<const int> yPred,
    const std::vector<std::string>& targetNames) {
    
    // Check for empty or mismatched input
    if (yTrue.empty() || yPred.empty() || yTrue.size() != yPred.size()) {
        return "Invalid input data";
    }
    
    // Identify unique classes through recursion
    const auto findUniqueClasses = [](auto& self, std::span<const int> labels, 
                                   size_t index, std::set<int>& classes) -> void {
        // Base case: all labels processed
        if (index >= labels.size()) {
            return;
        }
        
        // Add this label to the set and recurse
        classes.insert(labels[index]);
        self(self, labels, index + 1, classes);
    };
    
    std::set<int> classes;
    findUniqueClasses(findUniqueClasses, yTrue, 0, classes);
    
    // Check if we have enough target names
    if (targetNames.size() < classes.size()) {
        return "Not enough target names provided";
    }
    
    // Calculate metrics for each class through recursion
    std::unordered_map<int, double> precision;
    std::unordered_map<int, double> recall;
    std::unordered_map<int, double> f1;
    std::unordered_map<int, int> support;
    
    const auto calculateMetricsForClass = [&](auto& self, auto classIter) -> void {
        // Base case: all classes processed
        if (classIter == classes.end()) {
            return;
        }
        
        // Current class value
        int classVal = *classIter;
        
        // Count true positives, false positives, and false negatives
        const auto countMetrics = [](auto& self, std::span<const int> yTrue, 
                                 std::span<const int> yPred, int classVal, size_t index,
                                 int& tp, int& fp, int& fn) -> void {
            // Base case: all predictions processed
            if (index >= yTrue.size()) {
                return;
            }
            
            // Update counts based on this prediction
            if (yPred[index] == classVal && yTrue[index] == classVal) {
                tp++;
            } else if (yPred[index] == classVal && yTrue[index] != classVal) {
                fp++;
            } else if (yPred[index] != classVal && yTrue[index] == classVal) {
                fn++;
            }
            
            // Recursively process the next prediction
            self(self, yTrue, yPred, classVal, index + 1, tp, fp, fn);
        };
        
        int truePositives = 0;
        int falsePositives = 0;
        int falseNegatives = 0;
        
        countMetrics(countMetrics, yTrue, yPred, classVal, 0, 
                   truePositives, falsePositives, falseNegatives);
        
        // Calculate precision
        precision[classVal] = (truePositives + falsePositives > 0) ?
                             static_cast<double>(truePositives) / (truePositives + falsePositives) : 0.0;
        
        // Calculate recall
        recall[classVal] = (truePositives + falseNegatives > 0) ?
                          static_cast<double>(truePositives) / (truePositives + falseNegatives) : 0.0;
        
        // Calculate F1 score
        f1[classVal] = (precision[classVal] + recall[classVal] > 0) ?
                      2.0 * (precision[classVal] * recall[classVal]) / (precision[classVal] + recall[classVal]) : 0.0;
        
        // Calculate support (number of occurrences of each class)
        support[classVal] = std::count(yTrue.begin(), yTrue.end(), classVal);
        
        // Recursively process the next class
        self(self, std::next(classIter));
    };
    
    calculateMetricsForClass(calculateMetricsForClass, classes.begin());
    
    // Generate the report
    std::stringstream report;
    report << std::setw(20) << "precision" << std::setw(12) << "recall" 
           << std::setw(10) << "f1-score" << std::setw(10) << "support" << std::endl;
    report << std::string(52, '-') << std::endl;
    
    // Function to generate report rows recursively
    const auto generateReportRows = [&](auto& self, auto classIter, 
                                    double& totalPrecision, double& totalRecall, 
                                    double& totalF1, int& totalSupport, int& classCount) -> void {
        // Base case: all classes processed
        if (classIter == classes.end()) {
            return;
        }
        
        // Current class value
        int classVal = *classIter;
        
        // Find the position of this class in the classes set
        auto it = classes.find(classVal);
        size_t idx = std::distance(classes.begin(), it);
        
        // Write the row for this class
        report << std::setw(20) << targetNames[idx] 
               << std::setw(12) << std::fixed << std::setprecision(2) << precision[classVal]
               << std::setw(10) << std::fixed << std::setprecision(2) << recall[classVal]
               << std::setw(10) << std::fixed << std::setprecision(2) << f1[classVal]
               << std::setw(10) << support[classVal] << std::endl;
        
        // Update totals
        totalPrecision += precision[classVal];
        totalRecall += recall[classVal];
        totalF1 += f1[classVal];
        totalSupport += support[classVal];
        classCount++;
        
        // Recursively process the next class
        self(self, std::next(classIter), totalPrecision, totalRecall, totalF1, totalSupport, classCount);
    };
    
    double totalPrecision = 0.0;
    double totalRecall = 0.0;
    double totalF1 = 0.0;
    int totalSupport = 0;
    int classCount = 0;
    
    generateReportRows(generateReportRows, classes.begin(), 
                     totalPrecision, totalRecall, totalF1, totalSupport, classCount);
    
    // Calculate macro averages
    double avgPrecision = classCount > 0 ? totalPrecision / classCount : 0.0;
    double avgRecall = classCount > 0 ? totalRecall / classCount : 0.0;
    double avgF1 = classCount > 0 ? totalF1 / classCount : 0.0;
    
    // Add the averages row
    report << std::endl;
    report << std::setw(20) << "avg / total" 
           << std::setw(12) << std::fixed << std::setprecision(2) << avgPrecision
           << std::setw(10) << std::fixed << std::setprecision(2) << avgRecall
           << std::setw(10) << std::fixed << std::setprecision(2) << avgF1
           << std::setw(10) << totalSupport << std::endl;
    
    return report.str();
}

/**
 * Calculates classification performance metrics.
 * 
 * This function computes various metrics that evaluate classifier performance,
 * including accuracy, precision, recall, and F1-score.
 * 
 * @param yTrue Vector of true labels
 * @param yPred Vector of predicted labels
 * @return Map of metric names to values
 */
std::unordered_map<std::string, double> ModelEvaluator::calculateMetrics(
    std::span<const int> yTrue, std::span<const int> yPred) {
    
    // Check for empty or mismatched input
    if (yTrue.empty() || yPred.empty() || yTrue.size() != yPred.size()) {
        return {{"error", 1.0}};
    }
    
    // Calculate accuracy through recursion
    const auto calculateAccuracy = [](auto& self, std::span<const int> yTrue, 
                                   std::span<const int> yPred, size_t index, 
                                   size_t correct) -> double {
        // Base case: all predictions processed
        if (index >= yTrue.size()) {
            return static_cast<double>(correct) / yTrue.size();
        }
        
        // Update correct count if prediction matches actual
        size_t newCorrect = correct + (yTrue[index] == yPred[index] ? 1 : 0);
        
        // Recursively process the next prediction
        return self(self, yTrue, yPred, index + 1, newCorrect);
    };
    
    double accuracy = calculateAccuracy(calculateAccuracy, yTrue, yPred, 0, 0);
    
    // Identify unique classes
    const auto findUniqueClasses = [](auto& self, std::span<const int> labels, 
                                   size_t index, std::set<int>& classes) -> void {
        // Base case: all labels processed
        if (index >= labels.size()) {
            return;
        }
        
        // Add this label to the set and recurse
        classes.insert(labels[index]);
        self(self, labels, index + 1, classes);
    };
    
    std::set<int> classes;
    findUniqueClasses(findUniqueClasses, yTrue, 0, classes);
    findUniqueClasses(findUniqueClasses, yPred, 0, classes);
    
    // Calculate per-class metrics through recursion
    std::unordered_map<int, int> truePositives;
    std::unordered_map<int, int> falsePositives;
    std::unordered_map<int, int> falseNegatives;
    std::unordered_map<int, int> trueNegatives;
    
    const auto calculateClassMetrics = [&](auto& self, auto classIter) -> void {
        // Base case: all classes processed
        if (classIter == classes.end()) {
            return;
        }
        
        // Current class value
        int classVal = *classIter;
        
        // Count confusion matrix elements for this class
        const auto countElements = [](auto& self, std::span<const int> yTrue, 
                                  std::span<const int> yPred, int classVal, size_t index,
                                  int& tp, int& fp, int& fn, int& tn) -> void {
            // Base case: all predictions processed
            if (index >= yTrue.size()) {
                return;
            }
            
            // Update counts based on this prediction
            if (yTrue[index] == classVal && yPred[index] == classVal) {
                tp++;
            } else if (yTrue[index] != classVal && yPred[index] == classVal) {
                fp++;
            } else if (yTrue[index] == classVal && yPred[index] != classVal) {
                fn++;
            } else if (yTrue[index] != classVal && yPred[index] != classVal) {
                tn++;
            }
            
            // Recursively process the next prediction
            self(self, yTrue, yPred, classVal, index + 1, tp, fp, fn, tn);
        };
        
        int tp = 0, fp = 0, fn = 0, tn = 0;
        countElements(countElements, yTrue, yPred, classVal, 0, tp, fp, fn, tn);
        
        truePositives[classVal] = tp;
        falsePositives[classVal] = fp;
        falseNegatives[classVal] = fn;
        trueNegatives[classVal] = tn;
        
        // Recursively process the next class
        self(self, std::next(classIter));
    };
    
    calculateClassMetrics(calculateClassMetrics, classes.begin());
    
    // Calculate macro-averaged metrics
    double macroTp = 0.0, macroFp = 0.0, macroFn = 0.0, macroTn = 0.0;
    
    for (int classVal : classes) {
        macroTp += truePositives[classVal];
        macroFp += falsePositives[classVal];
        macroFn += falseNegatives[classVal];
        macroTn += trueNegatives[classVal];
    }
    
    macroTp /= classes.size();
    macroFp /= classes.size();
    macroFn /= classes.size();
    macroTn /= classes.size();
    
    // Calculate macro-averaged precision, recall, f1
    double macroPrecision = (macroTp + macroFp > 0) ? macroTp / (macroTp + macroFp) : 0.0;
    double macroRecall = (macroTp + macroFn > 0) ? macroTp / (macroTp + macroFn) : 0.0;
    double macroF1 = (macroPrecision + macroRecall > 0) ?
                   2.0 * (macroPrecision * macroRecall) / (macroPrecision + macroRecall) : 0.0;
    
    // Return all metrics
    return {
        {"accuracy", accuracy},
        {"macro_precision", macroPrecision},
        {"macro_recall", macroRecall},
        {"macro_f1", macroF1}
    };
}

} // namespace nlp

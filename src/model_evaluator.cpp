#include "sentiment_analysis.h"
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <set>

namespace nlp {

std::vector<std::vector<int>> ModelEvaluator::confusionMatrix(
    const std::vector<int>& yTrue, const std::vector<int>& yPred) {
    
    if (yTrue.size() != yPred.size() || yTrue.empty()) {
        return {{0}}; // Return empty matrix for invalid input
    }
    
    // Identify unique classes
    std::set<int> classes;
    for (auto label : yTrue) {
        classes.insert(label);
    }
    for (auto label : yPred) {
        classes.insert(label);
    }
    
    // Map class values to matrix indices
    std::unordered_map<int, size_t> classIndices;
    size_t idx = 0;
    for (int classVal : classes) {
        classIndices[classVal] = idx++;
    }
    
    // Initialize confusion matrix with zeros
    std::vector<std::vector<int>> matrix(classes.size(), std::vector<int>(classes.size(), 0));
    
    // Fill the confusion matrix
    for (size_t i = 0; i < yTrue.size(); ++i) {
        size_t trueIdx = classIndices[yTrue[i]];
        size_t predIdx = classIndices[yPred[i]];
        matrix[trueIdx][predIdx]++;
    }
    
    return matrix;
}

std::string ModelEvaluator::classificationReport(
    const std::vector<int>& yTrue, const std::vector<int>& yPred,
    const std::vector<std::string>& targetNames) {
    
    if (yTrue.size() != yPred.size() || yTrue.empty()) {
        return "Invalid input data";
    }
    
    // Identify unique classes
    std::set<int> classes;
    for (auto label : yTrue) {
        classes.insert(label);
    }
    
    // Make sure we have enough target names
    if (targetNames.size() < classes.size()) {
        return "Not enough target names provided";
    }
    
    // Calculate metrics for each class
    std::unordered_map<int, double> precision;
    std::unordered_map<int, double> recall;
    std::unordered_map<int, double> f1;
    std::unordered_map<int, int> support;
    
    for (int classVal : classes) {
        int truePositives = 0;
        int falsePositives = 0;
        int falseNegatives = 0;
        
        for (size_t i = 0; i < yTrue.size(); ++i) {
            if (yPred[i] == classVal && yTrue[i] == classVal) {
                truePositives++;
            } else if (yPred[i] == classVal && yTrue[i] != classVal) {
                falsePositives++;
            } else if (yPred[i] != classVal && yTrue[i] == classVal) {
                falseNegatives++;
            }
        }
        
        // Calculate precision
        precision[classVal] = (truePositives + falsePositives > 0) ?
                             static_cast<double>(truePositives) / (truePositives + falsePositives) : 0.0;
        
        // Calculate recall
        recall[classVal] = (truePositives + falseNegatives > 0) ?
                          static_cast<double>(truePositives) / (truePositives + falseNegatives) : 0.0;
        
        // Calculate F1 score
        f1[classVal] = (precision[classVal] + recall[classVal] > 0) ?
                      2.0 * (precision[classVal] * recall[classVal]) / (precision[classVal] + recall[classVal]) : 0.0;
        
        // Calculate support (number of occurrences of each class in yTrue)
        support[classVal] = std::count(yTrue.begin(), yTrue.end(), classVal);
    }
    
    // Generate the report
    std::stringstream report;
    report << std::setw(20) << "precision" << std::setw(12) << "recall" 
           << std::setw(10) << "f1-score" << std::setw(10) << "support" << std::endl;
    report << std::string(52, '-') << std::endl;
    
    double totalPrecision = 0.0;
    double totalRecall = 0.0;
    double totalF1 = 0.0;
    int totalSupport = 0;
    int classCount = 0;
    
    for (int classVal : classes) {
        size_t idx = std::distance(classes.begin(), classes.find(classVal));
        report << std::setw(20) << targetNames[idx] 
               << std::setw(12) << std::fixed << std::setprecision(2) << precision[classVal]
               << std::setw(10) << std::fixed << std::setprecision(2) << recall[classVal]
               << std::setw(10) << std::fixed << std::setprecision(2) << f1[classVal]
               << std::setw(10) << support[classVal] << std::endl;
        
        totalPrecision += precision[classVal];
        totalRecall += recall[classVal];
        totalF1 += f1[classVal];
        totalSupport += support[classVal];
        classCount++;
    }
    
    // Calculate averages
    double avgPrecision = totalPrecision / classCount;
    double avgRecall = totalRecall / classCount;
    double avgF1 = totalF1 / classCount;
    
    report << std::endl;
    report << std::setw(20) << "avg / total" 
           << std::setw(12) << std::fixed << std::setprecision(2) << avgPrecision
           << std::setw(10) << std::fixed << std::setprecision(2) << avgRecall
           << std::setw(10) << std::fixed << std::setprecision(2) << avgF1
           << std::setw(10) << totalSupport << std::endl;
    
    return report.str();
}

} // namespace nlp
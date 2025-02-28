/**
 * @file sgd_classifier.cpp
 * @brief Implementation of the SGDClassifier class
 * 
 * This file contains the implementation of the Stochastic Gradient Descent
 * classifier for sentiment analysis. The SGD algorithm is a simple yet
 * efficient optimization method for training linear classifiers.
 */

#include "sentiment_analysis.h"
#include <algorithm>
#include <random>
#include <cmath>
#include <numeric>
#include <iostream>
#include <ranges>

namespace nlp {

/**
 * Constructor initializes the classifier with configuration parameters.
 * 
 * @param loss Loss function to use
 * @param learningRate Learning rate schedule
 * @param penalty Regularization type
 * @param alpha Regularization strength
 * @param eta0 Initial learning rate
 */
SGDClassifier::SGDClassifier(const std::string& loss, 
                           const std::string& learningRate,
                           const std::string& penalty,
                           double alpha, 
                           double eta0)
    : loss(loss), learningRate(learningRate), penalty(penalty),
      alpha(alpha), eta0(eta0), intercept(0.0) {
}

/**
 * Trains the classifier on training data using SGD algorithm.
 * 
 * This method implements the Stochastic Gradient Descent algorithm
 * for training a binary classifier, using a functional approach.
 * 
 * @param X Training feature matrix
 * @param y Training labels
 */
void SGDClassifier::fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y) {
    // Don't train if no data
    if (X.empty() || y.empty() || X[0].empty()) {
        std::cerr << "Error: Empty training data" << std::endl;
        return;
    }
    
    // Number of features
    const size_t nFeatures = X[0].size();
    
    // Initialize weights with small random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(0.0, 0.01);
    
    weights.resize(nFeatures);
    std::ranges::generate(weights, [&dist, &gen]() { return dist(gen); });
    intercept = 0.0;
    
    // Count class frequencies
    std::unordered_map<int, size_t> classCounts;
    for (int label : y) {
        classCounts[label]++;
    }
    
    // Compute class weights inversely proportional to frequencies
    std::unordered_map<int, double> classWeights;
    for (const auto& [label, count] : classCounts) {
        classWeights[label] = static_cast<double>(y.size()) / (classCounts.size() * count);
    }
    
    // Print class distribution and weights
    std::cout << "Class distribution in training data:" << std::endl;
    for (const auto& [label, count] : classCounts) {
        std::cout << "Class " << label << ": " << count << " samples, weight: "
                 << classWeights[label] << std::endl;
    }
    
    // Convert sentiment labels to binary (-1/1) for SGD
    std::vector<int> binaryY(y.size());
    std::ranges::transform(y, binaryY.begin(), 
                        [](int label) { return (label == 0) ? -1 : 1; });
    
    // Number of epochs for training
    const int nEpochs = 10;
    
    // Create indices for shuffling
    std::vector<size_t> indices(X.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    // Use a smaller initial learning rate for stability
    const double initialEta = 0.001;
    
    // Training loop for epochs
    for (int epoch = 0; epoch < nEpochs; ++epoch) {
        // Create a copy of indices for shuffling
        auto currIndices = indices;
        
        // Shuffle indices
        std::shuffle(currIndices.begin(), currIndices.end(), gen);
        
        // Calculate learning rate for this epoch
        double eta = (learningRate == "adaptive") ?
                     initialEta / (1.0 + alpha * initialEta * static_cast<double>(epoch)) :
                     initialEta;
        
        // Track statistics for this epoch
        int misclassified = 0;
        double totalLoss = 0.0;
        
        // Process each sample
        for (size_t idx : currIndices) {
            const auto& xi = X[idx];
            const int target = binaryY[idx];
            
            // Skip if feature vector is empty
            if (xi.empty()) {
                continue;
            }
            
            // Calculate prediction using dot product
            double prediction = 0.0;
            for (size_t j = 0; j < std::min(weights.size(), xi.size()); ++j) {
                prediction += weights[j] * xi[j];
            }
            prediction += intercept;
            
            // Apply class weight to learning rate
            const double sampleEta = eta * classWeights[y[idx]];
            
            // Compute updates based on loss function
            if (loss == "modified_huber") {
                // Compute loss for this sample
                double sampleLoss = 0.0;
                if (target * prediction < -1.0) {
                    sampleLoss = -4.0 * target * prediction;
                } else if (target * prediction < 1.0) {
                    sampleLoss = (1.0 - target * prediction) * (1.0 - target * prediction);
                }
                totalLoss += sampleLoss;
                
                // Update weights if margin is not satisfied
                if (target * prediction < 1.0) {
                    // Calculate update multiplier
                    const double multiplier = (target * prediction < -1.0) ? 
                                          -target : target * (1.0 - target * prediction);
                    
                    // Update weights with gradient and regularization
                    for (size_t j = 0; j < weights.size(); ++j) {
                        if (j < xi.size()) {
                            weights[j] = (1.0 - sampleEta * alpha) * weights[j] +
                                       sampleEta * multiplier * xi[j];
                        } else {
                            // Just apply regularization
                            weights[j] *= (1.0 - sampleEta * alpha);
                        }
                    }
                    
                    // Update intercept (no regularization on intercept)
                    intercept += sampleEta * multiplier;
                    
                    // Count misclassifications
                    if (target * prediction < 0) {
                        misclassified++;
                    }
                } else {
                    // Apply regularization only if margin is satisfied
                    for (size_t j = 0; j < weights.size(); ++j) {
                        weights[j] *= (1.0 - sampleEta * alpha);
                    }
                }
            } else {
                // Hinge loss (SVM)
                if (target * prediction < 1.0) {
                    // Update weights with gradient and regularization
                    for (size_t j = 0; j < weights.size(); ++j) {
                        if (j < xi.size()) {
                            weights[j] = (1.0 - sampleEta * alpha) * weights[j] +
                                       sampleEta * target * xi[j];
                        } else {
                            // Just apply regularization
                            weights[j] *= (1.0 - sampleEta * alpha);
                        }
                    }
                    
                    // Update intercept
                    intercept += sampleEta * target;
                    
                    // Count misclassifications
                    if (target * prediction < 0) {
                        misclassified++;
                    }
                } else {
                    // Apply regularization only
                    for (size_t j = 0; j < weights.size(); ++j) {
                        weights[j] *= (1.0 - sampleEta * alpha);
                    }
                }
            }
        }
        
        // Print training progress
        const double errorRate = static_cast<double>(misclassified) / y.size();
        const double avgLoss = totalLoss / y.size();
        
        std::cout << "Epoch " << (epoch + 1) << "/" << nEpochs
                 << ", Misclassified: " << misclassified
                 << ", Error rate: " << errorRate
                 << ", Average loss: " << avgLoss
                 << std::endl;
    }
    
    // Print weight statistics
    const double sum = std::accumulate(weights.begin(), weights.end(), 0.0);
    const double avgWeight = sum / weights.size();
    
    const double absSum = std::accumulate(weights.begin(), weights.end(), 0.0,
                                     [](double acc, double w) { return acc + std::abs(w); });
    const double avgAbsWeight = absSum / weights.size();
    
    // Find max absolute weight
    const auto maxAbsElem = std::max_element(weights.begin(), weights.end(),
                                       [](double a, double b) { return std::abs(a) < std::abs(b); });
    const double maxAbsWeight = (maxAbsElem != weights.end()) ? std::abs(*maxAbsElem) : 0.0;
    
    // Count nonzero weights
    const size_t nonzeroCount = std::count_if(weights.begin(), weights.end(),
                                          [](double w) { return std::abs(w) > 1e-5; });
    
    std::cout << "Weight statistics:" << std::endl;
    std::cout << "Average weight: " << avgWeight << std::endl;
    std::cout << "Average absolute weight: " << avgAbsWeight << std::endl;
    std::cout << "Max absolute weight: " << maxAbsWeight << std::endl;
    std::cout << "Nonzero weights: " << nonzeroCount << " out of " << weights.size() << std::endl;
    std::cout << "Intercept (bias): " << intercept << std::endl;
}

/**
 * Predicts sentiment labels for new data.
 * 
 * This method applies the trained classifier to new feature vectors
 * to predict sentiment labels (0=negative, 4=positive).
 * 
 * @param X Feature matrix to predict
 * @return Vector of predicted labels
 */
std::vector<int> SGDClassifier::predict(const std::vector<std::vector<double>>& X) const {
    std::vector<int> predictions;
    predictions.reserve(X.size());
    
    for (const auto& x : X) {
        // Calculate decision value
        double decision = 0.0;
        
        for (size_t i = 0; i < std::min(weights.size(), x.size()); ++i) {
            decision += weights[i] * x[i];
        }
        decision += intercept;
        
        // Convert to sentiment class (0=negative, 4=positive)
        predictions.push_back(decision >= 0.0 ? 4 : 0);
    }
    
    return predictions;
}

/**
 * Calculates decision function values for new data.
 * 
 * The decision function gives the raw score for each sample, which
 * indicates the confidence and direction of the prediction.
 * 
 * @param X Feature matrix
 * @return Vector of decision function values
 */
std::vector<double> SGDClassifier::decisionFunction(const std::vector<std::vector<double>>& X) const {
    std::vector<double> decisions;
    decisions.reserve(X.size());
    
    for (const auto& x : X) {
        // Calculate decision value
        double decision = 0.0;
        
        for (size_t i = 0; i < std::min(weights.size(), x.size()); ++i) {
            decision += weights[i] * x[i];
        }
        decision += intercept;
        
        // Apply scaling to prevent extreme values
        if (std::abs(decision) > 10.0) {
            decision = std::copysign(10.0, decision);
        }
        
        decisions.push_back(decision);
    }
    
    return decisions;
}

/**
 * Calculates the prediction accuracy on test data.
 * 
 * @param X Test feature matrix
 * @param y Test labels
 * @return Accuracy score between 0 and 1
 */
double SGDClassifier::score(const std::vector<std::vector<double>>& X, const std::vector<int>& y) const {
    // Check for empty input
    if (X.empty() || y.empty() || X.size() != y.size()) {
        return 0.0;
    }
    
    // Get predictions
    std::vector<int> predictions = predict(X);
    
    // Count correct predictions
    size_t correct = 0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        if (predictions[i] == y[i]) {
            correct++;
        }
    }
    
    // Return accuracy as proportion of correct predictions
    return static_cast<double>(correct) / predictions.size();
}

/**
 * Converts a decision value to a probability estimate.
 * 
 * @param x Feature vector
 * @return Probability estimate between 0 and 1
 */
double SGDClassifier::predict_proba(const std::vector<double>& x) const {
    // Calculate decision value
    double decision = 0.0;
    
    for (size_t i = 0; i < std::min(weights.size(), x.size()); ++i) {
        decision += weights[i] * x[i];
    }
    decision += intercept;
    
    // Apply scaling
    if (std::abs(decision) > 10.0) {
        decision = std::copysign(10.0, decision);
    }
    
    // Convert to probability with sigmoid function
    if (loss == "modified_huber") {
        // Modified Huber loss gives calibrated probabilities
        if (decision >= 1.0) {
            return 1.0;
        } else if (decision <= -1.0) {
            return 0.0;
        } else {
            return (decision + 1.0) / 2.0;
        }
    } else {
        // Standard sigmoid function for other loss functions
        return 1.0 / (1.0 + std::exp(-decision));
    }
}

} // namespace nlp

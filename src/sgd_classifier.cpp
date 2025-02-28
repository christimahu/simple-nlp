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
 * for training a binary classifier, using a functional approach with
 * recursion rather than traditional loops.
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
    auto initializeWeights = [nFeatures]() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> dist(0.0, 0.01);
        
        std::vector<double> weights(nFeatures);
        std::ranges::generate(weights, [&dist, &gen]() { return dist(gen); });
        return weights;
    };
    
    weights = initializeWeights();
    intercept = 0.0;
    
    // Count class frequencies
    auto countClasses = [](const std::vector<int>& labels) {
        std::unordered_map<int, size_t> counts;
        for (int label : labels) {
            counts[label]++;
        }
        return counts;
    };
    
    auto classCounts = countClasses(y);
    
    // Compute class weights inversely proportional to frequencies
    auto computeClassWeights = [&y, &classCounts]() {
        std::unordered_map<int, double> weights;
        for (const auto& [label, count] : classCounts) {
            weights[label] = static_cast<double>(y.size()) / (classCounts.size() * count);
        }
        return weights;
    };
    
    auto classWeights = computeClassWeights();
    
    // Print class distribution and weights
    std::cout << "Class distribution in training data:" << std::endl;
    for (const auto& [label, count] : classCounts) {
        std::cout << "Class " << label << ": " << count << " samples, weight: "
                 << classWeights[label] << std::endl;
    }
    
    // Convert sentiment labels to binary (-1/1) for SGD
    auto convertLabels = [](const std::vector<int>& labels) {
        std::vector<int> binaryLabels(labels.size());
        std::ranges::transform(labels, binaryLabels.begin(), 
                            [](int label) { return (label == 0) ? -1 : 1; });
        return binaryLabels;
    };
    
    auto binaryY = convertLabels(y);
    
    // Number of epochs for training
    const int nEpochs = 10;
    
    // Create indices for shuffling
    std::vector<size_t> indices(X.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    // Use a smaller initial learning rate for stability
    const double initialEta = 0.001;
    
    // Recursive function to process one epoch
    const auto processEpoch = [&](auto& self, int epoch, const std::vector<size_t>& shuffledIndices) -> void {
        // Base case: all epochs processed
        if (epoch >= nEpochs) {
            return;
        }
        
        // Create a copy of indices for shuffling
        auto currIndices = shuffledIndices;
        
        // Shuffle indices
        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(currIndices.begin(), currIndices.end(), gen);
        
        // Calculate learning rate for this epoch
        double eta = (learningRate == "adaptive") ?
                     initialEta / (1.0 + alpha * initialEta * static_cast<double>(epoch)) :
                     initialEta;
        
        // Recursive function to process one sample
        const auto processSample = [&](auto& self, size_t index, int& misclassified, double& totalLoss) -> void {
            // Base case: all samples processed
            if (index >= currIndices.size()) {
                return;
            }
            
            const size_t idx = currIndices[index];
            const auto& xi = X[idx];
            const int target = binaryY[idx];
            
            // Skip if feature vector is empty
            if (xi.empty()) {
                self(self, index + 1, misclassified, totalLoss);
                return;
            }
            
            // Calculate prediction using dot product
            const auto dotProduct = [](const auto& a, const auto& b, double bias) {
                // Base case for recursive dot product
                const auto dotProductHelper = [](auto& self, const auto& a, const auto& b, 
                                             size_t index, double sum) -> double {
                    // Base case: all elements processed
                    if (index >= a.size() || index >= b.size()) {
                        return sum;
                    }
                    
                    // Add product of current elements to sum and recurse
                    return self(self, a, b, index + 1, sum + a[index] * b[index]);
                };
                
                return dotProductHelper(dotProductHelper, a, b, 0, bias);
            };
            
            const double prediction = dotProduct(xi, weights, intercept);
            
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
                    
                    // Recursive function to update weights
                    const auto updateWeights = [&](auto& self, size_t idx) -> void {
                        // Base case: all weights updated
                        if (idx >= weights.size()) {
                            return;
                        }
                        
                        // Skip if feature index is out of bounds
                        if (idx >= xi.size()) {
                            // Just apply regularization
                            weights[idx] *= (1.0 - sampleEta * alpha);
                        } else {
                            // Update with gradient and regularization
                            weights[idx] = (1.0 - sampleEta * alpha) * weights[idx] +
                                         sampleEta * multiplier * xi[idx];
                        }
                        
                        // Recursively update next weight
                        self(self, idx + 1);
                    };
                    
                    // Start the recursive weight update
                    updateWeights(updateWeights, 0);
                    
                    // Update intercept (no regularization on intercept)
                    intercept += sampleEta * multiplier;
                    
                    // Count misclassifications
                    if (target * prediction < 0) {
                        misclassified++;
                    }
                } else {
                    // Apply regularization only if margin is satisfied
                    // Recursive function to apply regularization
                    const auto applyRegularization = [&](auto& self, size_t idx) -> void {
                        // Base case: all weights updated
                        if (idx >= weights.size()) {
                            return;
                        }
                        
                        // Apply weight decay
                        weights[idx] *= (1.0 - sampleEta * alpha);
                        
                        // Recursively update next weight
                        self(self, idx + 1);
                    };
                    
                    // Start the recursive regularization
                    applyRegularization(applyRegularization, 0);
                }
            } else {
                // Hinge loss (SVM)
                if (target * prediction < 1.0) {
                    // Recursive function to update weights
                    const auto updateWeights = [&](auto& self, size_t idx) -> void {
                        // Base case: all weights updated
                        if (idx >= weights.size()) {
                            return;
                        }
                        
                        // Skip if feature index is out of bounds
                        if (idx >= xi.size()) {
                            // Just apply regularization
                            weights[idx] *= (1.0 - sampleEta * alpha);
                        } else {
                            // Update with gradient and regularization
                            weights[idx] = (1.0 - sampleEta * alpha) * weights[idx] +
                                         sampleEta * target * xi[idx];
                        }
                        
                        // Recursively update next weight
                        self(self, idx + 1);
                    };
                    
                    // Start the recursive weight update
                    updateWeights(updateWeights, 0);
                    
                    // Update intercept
                    intercept += sampleEta * target;
                    
                    // Count misclassifications
                    if (target * prediction < 0) {
                        misclassified++;
                    }
                } else {
                    // Apply regularization only
                    // Recursive function to apply regularization
                    const auto applyRegularization = [&](auto& self, size_t idx) -> void {
                        // Base case: all weights updated
                        if (idx >= weights.size()) {
                            return;
                        }
                        
                        // Apply weight decay
                        weights[idx] *= (1.0 - sampleEta * alpha);
                        
                        // Recursively update next weight
                        self(self, idx + 1);
                    };
                    
                    // Start the recursive regularization
                    applyRegularization(applyRegularization, 0);
                }
            }
            
            // Process next sample
            self(self, index + 1, misclassified, totalLoss);
        };
        
        // Process all samples in this epoch
        int misclassified = 0;
        double totalLoss = 0.0;
        processSample(processSample, 0, misclassified, totalLoss);
        
        // Print training progress
        const double errorRate = static_cast<double>(misclassified) / y.size();
        const double avgLoss = totalLoss / y.size();
        
        std::cout << "Epoch " << (epoch + 1) << "/" << nEpochs
                 << ", Misclassified: " << misclassified
                 << ", Error rate: " << errorRate
                 << ", Average loss: " << avgLoss
                 << std::endl;
        
        // Process next epoch
        self(self, epoch + 1, currIndices);
    };
    
    // Start the recursive epoch processing
    processEpoch(processEpoch, 0, indices);
    
    // Print weight statistics
    auto weightStats = [&weights]() {
        // Calculate stats using STL algorithms
        const double sum = std::accumulate(weights.begin(), weights.end(), 0.0);
        const double absSum = std::accumulate(weights.begin(), weights.end(), 0.0,
                                         [](double acc, double w) { return acc + std::abs(w); });
        
        // Find max absolute weight
        const auto maxAbsElem = std::max_element(weights.begin(), weights.end(),
                                           [](double a, double b) { return std::abs(a) < std::abs(b); });
        const double maxAbs = (maxAbsElem != weights.end()) ? std::abs(*maxAbsElem) : 0.0;
        
        // Count nonzero weights
        const size_t nonzeroCount = std::count_if(weights.begin(), weights.end(),
                                              [](double w) { return std::abs(w) > 1e-5; });
        
        return std::make_tuple(sum / weights.size(), absSum / weights.size(), maxAbs, nonzeroCount);
    };
    
    auto [avgWeight, avgAbsWeight, maxAbsWeight, nonzeroCount] = weightStats();
    
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
    // Recursive approach to prediction
    const auto predictRecursive = [this](auto& self, const auto& X, size_t index, std::vector<int>& predictions) -> void {
        // Base case: all samples processed
        if (index >= X.size()) {
            return;
        }
        
        const auto& x = X[index];
        
        // Skip if feature vector is empty
        if (x.empty()) {
            predictions.push_back(0);  // Default to negative class
            self(self, X, index + 1, predictions);
            return;
        }
        
        // Calculate decision value
        double decision = 0.0;
        
        // Calculate the dot product recursively
        const auto calculateDecision = [&](auto& self, size_t featureIdx, double sum) -> double {
            // Base case: all features processed
            if (featureIdx >= weights.size() || featureIdx >= x.size()) {
                return sum + intercept;
            }
            
            // Add this feature's contribution and recurse
            return self(self, featureIdx + 1, sum + weights[featureIdx] * x[featureIdx]);
        };
        
        decision = calculateDecision(calculateDecision, 0, 0.0);
        
        // Convert to sentiment class (0=negative, 4=positive)
        predictions.push_back(decision >= 0.0 ? 4 : 0);
        
        // Process next sample
        self(self, X, index + 1, predictions);
    };
    
    std::vector<int> predictions;
    predictions.reserve(X.size());
    predictRecursive(predictRecursive, X, 0, predictions);
    
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
    // Recursive approach to calculating decision values
    const auto decisionRecursive = [this](auto& self, const auto& X, size_t index, std::vector<double>& decisions) -> void {
        // Base case: all samples processed
        if (index >= X.size()) {
            return;
        }
        
        const auto& x = X[index];
        
        // Skip if feature vector is empty
        if (x.empty()) {
            decisions.push_back(0.0);
            self(self, X, index + 1, decisions);
            return;
        }
        
        // Calculate decision value
        double decision = 0.0;
        
        // Calculate the dot product recursively
        const auto calculateDecision = [&](auto& self, size_t featureIdx, double sum) -> double {
            // Base case: all features processed
            if (featureIdx >= weights.size() || featureIdx >= x.size()) {
                return sum + intercept;
            }
            
            // Add this feature's contribution and recurse
            return self(self, featureIdx + 1, sum + weights[featureIdx] * x[featureIdx]);
        };
        
        decision = calculateDecision(calculateDecision, 0, 0.0);
        
        // Apply scaling to prevent extreme values
        if (std::abs(decision) > 10.0) {
            decision = std::copysign(10.0, decision);
        }
        
        decisions.push_back(decision);
        
        // Process next sample
        self(self, X, index + 1, decisions);
    };
    
    std::vector<double> decisions;
    decisions.reserve(X.size());
    decisionRecursive(decisionRecursive, X, 0, decisions);
    
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
    
    // Recursive approach to counting correct predictions
    const auto countCorrect = [](auto& self, const auto& pred, const auto& actual, 
                              size_t index, size_t correct) -> size_t {
        // Base case: all predictions checked
        if (index >= pred.size()) {
            return correct;
        }
        
        // Increment correct count if prediction matches actual
        size_t newCorrect = correct + (pred[index] == actual[index] ? 1 : 0);
        
        // Recursively check next prediction
        return self(self, pred, actual, index + 1, newCorrect);
    };
    
    size_t correct = countCorrect(countCorrect, predictions, y, 0, 0);
    
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
    
    // Calculate the dot product recursively
    const auto calculateDecision = [this](auto& self, const auto& x, 
                                      size_t featureIdx, double sum) -> double {
        // Base case: all features processed
        if (featureIdx >= weights.size() || featureIdx >= x.size()) {
            return sum + intercept;
        }
        
        // Add this feature's contribution and recurse
        return self(self, x, featureIdx + 1, sum + weights[featureIdx] * x[featureIdx]);
    };
    
    decision = calculateDecision(calculateDecision, x, 0, 0.0);
    
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

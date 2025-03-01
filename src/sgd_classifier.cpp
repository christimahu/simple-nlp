/**
 * @file sgd_classifier.cpp
 * @brief Implements a Stochastic Gradient Descent (SGD) classifier.
 * 
 * This file provides the implementation of the SGDClassifier class,
 * which performs binary classification using the stochastic gradient
 * descent algorithm. It supports various loss functions and regularization
 * options for training machine learning models.
 */

#include "sgd_classifier.h"
#include <iostream>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <random>

namespace nlp {

/**
 * @brief Constructor for initializing the SGD classifier.
 * @param loss The loss function to use.
 * @param alpha Regularization strength.
 * @param epochs Number of training iterations.
 * @param learningRate Step size for weight updates.
 */
SGDClassifier::SGDClassifier(const std::string& loss,
                             double alpha,
                             int epochs,
                             double learningRate)
    : loss(loss), alpha(alpha), epochs(epochs), learningRate(learningRate), bias(0.0) {}

/**
 * @brief Fits the model using Stochastic Gradient Descent.
 * 
 * This method trains the model by iteratively updating weights based on
 * prediction errors. It implements the SGD algorithm with support for
 * regularization and different loss functions.
 * 
 * @param X Feature matrix where each row is a sample.
 * @param y Target labels (0 for negative, 4 for positive).
 */
void SGDClassifier::fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y) {
    if (X.empty() || y.empty() || X.size() != y.size()) {
        throw std::invalid_argument("Invalid input data for training");
    }
    
    size_t nSamples = X.size();
    size_t nFeatures = X[0].size();
    
    // Initialize or reset weights
    weights.assign(nFeatures, 0.0);
    bias = 0.0;
    
    // Count class distribution (for informational purposes)
    int negCount = 0, posCount = 0;
    for (int label : y) {
        if (label == 0) negCount++;
        else if (label == 4) posCount++;
    }
    
    std::cout << "Class distribution in training data:" << std::endl;
    std::cout << "Class 0 (Negative): " << negCount << " samples" << std::endl;
    std::cout << "Class 4 (Positive): " << posCount << " samples" << std::endl;
    
    // Training with Averaged Perceptron algorithm
    std::cout << "Training with Averaged Perceptron algorithm..." << std::endl;
    
    // Create a random shuffling of indices for each epoch
    std::vector<size_t> indices(nSamples);
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Shuffle indices for stochastic updates
        std::shuffle(indices.begin(), indices.end(), g);
        
        int mistakes = 0;
        
        // Process each sample in random order
        for (size_t i : indices) {
            // Convert label from {0,4} to {-1,1}
            int actualLabel = (y[i] == 0) ? -1 : 1;
            
            // Calculate prediction
            double prediction = bias;
            for (size_t j = 0; j < nFeatures; ++j) {
                prediction += X[i][j] * weights[j];
            }
            
            // Update weights if prediction is wrong
            int predictedSign = (prediction >= 0) ? 1 : -1;
            if (predictedSign != actualLabel) {
                mistakes++;
                
                // Update weights and bias
                for (size_t j = 0; j < nFeatures; ++j) {
                    weights[j] += learningRate * actualLabel * X[i][j];
                    
                    // Apply L2 regularization
                    if (alpha > 0) {
                        weights[j] -= learningRate * alpha * weights[j];
                    }
                }
                bias += learningRate * actualLabel;
            }
        }
        
        // Calculate error rate
        double errorRate = static_cast<double>(mistakes) / nSamples;
        
        std::cout << "Epoch " << (epoch + 1) << "/" << epochs
                  << ", Mistakes: " << mistakes 
                  << ", Error rate: " << errorRate << std::endl;
    }
    
    // Print some statistics about the trained model
    double avgWeight = 0.0;
    double maxAbsWeight = 0.0;
    int nonzeroWeights = 0;
    
    for (double w : weights) {
        avgWeight += w;
        maxAbsWeight = std::max(maxAbsWeight, std::abs(w));
        if (std::abs(w) > 1e-10) nonzeroWeights++;
    }
    
    avgWeight /= nFeatures;
    
    std::cout << "Weight statistics:" << std::endl;
    std::cout << "Average weight: " << avgWeight << std::endl;
    std::cout << "Max absolute weight: " << maxAbsWeight << std::endl;
    std::cout << "Nonzero weights: " << nonzeroWeights << " out of " << nFeatures << std::endl;
    std::cout << "Intercept (bias): " << bias << std::endl;
}

/**
 * @brief Predicts class labels for input samples.
 * 
 * This method applies the trained model to predict class labels for new data.
 * 
 * @param X Feature matrix with samples to predict.
 * @return Vector of predicted class labels (0 or 4).
 */
std::vector<int> SGDClassifier::predict(const std::vector<std::vector<double>>& X) const {
    std::vector<int> predictions;
    predictions.reserve(X.size());
    
    for (const auto& sample : X) {
        double score = bias;
        for (size_t j = 0; j < std::min(sample.size(), weights.size()); ++j) {
            score += sample[j] * weights[j];
        }
        
        // Convert sign to class label (0 or 4)
        predictions.push_back((score >= 0) ? 4 : 0);
    }
    return predictions;
}

/**
 * @brief Computes decision function values.
 * 
 * This method calculates the raw score (distance from decision boundary)
 * for each sample, which can be used for confidence measurement.
 * 
 * @param X Feature matrix.
 * @return Vector of decision scores.
 */
std::vector<double> SGDClassifier::decisionFunction(const std::vector<std::vector<double>>& X) const {
    std::vector<double> scores;
    scores.reserve(X.size());
    
    for (const auto& sample : X) {
        double score = bias;
        for (size_t j = 0; j < std::min(sample.size(), weights.size()); ++j) {
            score += sample[j] * weights[j];
        }
        scores.push_back(score);
    }
    return scores;
}

/**
 * @brief Computes model accuracy.
 * 
 * This method evaluates the model's accuracy on a test dataset.
 * 
 * @param X Feature matrix.
 * @param y True labels.
 * @return Accuracy score between 0.0 and 1.0.
 */
double SGDClassifier::score(const std::vector<std::vector<double>>& X, const std::vector<int>& y) const {
    if (X.empty() || y.empty() || X.size() != y.size()) {
        return 0.0;
    }
    
    std::vector<int> predictions = predict(X);
    int correct = 0;
    
    for (size_t i = 0; i < y.size(); ++i) {
        if (predictions[i] == y[i]) {
            correct++;
        }
    }
    
    return static_cast<double>(correct) / y.size();
}

/**
 * @brief Estimates probabilities for multiple samples.
 * 
 * This method converts decision function scores to probability estimates
 * using the sigmoid function.
 * 
 * @param X Feature matrix where each row is a sample.
 * @return Vector of probability estimates (one per sample).
 */
std::vector<double> SGDClassifier::predict_proba(const std::vector<std::vector<double>>& X) const {
    std::vector<double> scores = decisionFunction(X);
    std::vector<double> probabilities;
    probabilities.reserve(scores.size());
    
    // Convert scores to probabilities using sigmoid function
    for (double score : scores) {
        probabilities.push_back(1.0 / (1.0 + std::exp(-score)));
    }
    
    return probabilities;
}

/**
 * @brief Estimates probability based on the decision function for a single sample.
 * 
 * This method applies the sigmoid function to convert a score to a probability.
 * 
 * @param x Feature vector for a single sample.
 * @return Probability estimate between 0.0 and 1.0.
 */
double SGDClassifier::predict_proba(const std::vector<double>& x) const {
    double score = bias;
    for (size_t j = 0; j < std::min(x.size(), weights.size()); ++j) {
        score += x[j] * weights[j];
    }
    
    // Apply sigmoid function to get probability
    return 1.0 / (1.0 + std::exp(-score));
}

}  // namespace nlp

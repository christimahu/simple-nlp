#include "sentiment_analysis.h"
#include <algorithm>
#include <random>
#include <cmath>
#include <numeric>
#include <iostream>

namespace nlp {

SGDClassifier::SGDClassifier(const std::string& loss, 
                           const std::string& learningRate,
                           const std::string& penalty,
                           double alpha, 
                           double eta0)
    : loss(loss), learningRate(learningRate), penalty(penalty),
      alpha(alpha), eta0(eta0), intercept(0.0) {
}

void SGDClassifier::fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y) {
    // Don't train if no data
    if (X.empty() || y.empty() || X[0].empty()) {
        std::cerr << "Error: Empty training data" << std::endl;
        return;
    }
    
    // Number of features
    const size_t n_features = X[0].size();
    
    // Initialize weights to zeros
    weights.resize(n_features, 0.0);
    intercept = 0.0;
    
    // Count class frequencies
    size_t neg_count = 0, pos_count = 0;
    for (int label : y) {
        if (label == 0) neg_count++;
        else pos_count++;
    }
    
    std::cout << "Class distribution in training data:" << std::endl;
    std::cout << "Class 0 (Negative): " << neg_count << " samples" << std::endl;
    std::cout << "Class 4 (Positive): " << pos_count << " samples" << std::endl;
    
    // Use simplified averaged perceptron algorithm instead of SGD
    // This is much more stable for text classification
    std::cout << "Training with Averaged Perceptron algorithm..." << std::endl;
    
    const int n_epochs = 5;
    
    // Create indices for iteration
    std::vector<size_t> indices(X.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    // For averaging
    std::vector<double> total_weights(n_features, 0.0);
    double total_intercept = 0.0;
    int n_updates = 0;
    
    // Create random generator for shuffling
    std::random_device rd;
    std::mt19937 g(rd());
    
    // Training loop
    for (int epoch = 0; epoch < n_epochs; ++epoch) {
        // Shuffle data
        std::shuffle(indices.begin(), indices.end(), g);
        
        int mistakes = 0;
        
        // Process each sample
        for (size_t i : indices) {
            // Convert sentiment label to -1/+1
            int target = (y[i] == 0) ? -1 : 1;
            
            // Calculate prediction
            double prediction = 0.0;
            const auto& xi = X[i];
            
            for (size_t j = 0; j < n_features; ++j) {
                if (j < xi.size()) { // Safety check
                    prediction += weights[j] * xi[j];
                }
            }
            prediction += intercept;
            
            // Update on mistake
            if (prediction * target <= 0) {
                // Very small constant learning rate
                const double eta = 0.01;
                
                // Update weights
                for (size_t j = 0; j < n_features; ++j) {
                    if (j < xi.size()) { // Safety check
                        weights[j] += eta * target * xi[j];
                    }
                }
                
                // Update intercept (bias)
                intercept += eta * target;
                
                // Count mistake
                mistakes++;
            }
            
            // Accumulate weights for averaging
            for (size_t j = 0; j < n_features; ++j) {
                total_weights[j] += weights[j];
            }
            total_intercept += intercept;
            n_updates++;
        }
        
        // Report progress
        double error_rate = static_cast<double>(mistakes) / X.size();
        std::cout << "Epoch " << (epoch + 1) << "/" << n_epochs
                  << ", Mistakes: " << mistakes
                  << ", Error rate: " << error_rate
                  << std::endl;
    }
    
    // Set final weights to the average
    if (n_updates > 0) {
        for (size_t j = 0; j < n_features; ++j) {
            weights[j] = total_weights[j] / n_updates;
        }
        intercept = total_intercept / n_updates;
    }
    
    // Print weight statistics
    double avg_weight = 0.0;
    double max_abs_weight = 0.0;
    size_t nonzero_count = 0;
    
    for (double w : weights) {
        avg_weight += w;
        max_abs_weight = std::max(max_abs_weight, std::abs(w));
        if (std::abs(w) > 1e-5) {
            nonzero_count++;
        }
    }
    
    avg_weight /= weights.size();
    
    std::cout << "Weight statistics:" << std::endl;
    std::cout << "Average weight: " << avg_weight << std::endl;
    std::cout << "Max absolute weight: " << max_abs_weight << std::endl;
    std::cout << "Nonzero weights: " << nonzero_count << " out of " << weights.size() << std::endl;
    std::cout << "Intercept (bias): " << intercept << std::endl;
}

std::vector<int> SGDClassifier::predict(const std::vector<std::vector<double>>& X) {
    std::vector<int> predictions;
    predictions.reserve(X.size());
    
    for (const auto& x : X) {
        // Calculate decision value
        double decision = 0.0;
        
        for (size_t i = 0; i < weights.size() && i < x.size(); ++i) {
            decision += weights[i] * x[i];
        }
        decision += intercept;
        
        // Convert to sentiment label
        predictions.push_back(decision >= 0.0 ? 4 : 0);
    }
    
    return predictions;
}

std::vector<double> SGDClassifier::decisionFunction(const std::vector<std::vector<double>>& X) {
    std::vector<double> decisions;
    decisions.reserve(X.size());
    
    for (const auto& x : X) {
        // Calculate decision value
        double decision = 0.0;
        
        for (size_t i = 0; i < weights.size() && i < x.size(); ++i) {
            decision += weights[i] * x[i];
        }
        decision += intercept;
        
        decisions.push_back(decision);
    }
    
    return decisions;
}

double SGDClassifier::score(const std::vector<std::vector<double>>& X, const std::vector<int>& y) {
    // Predict classes
    std::vector<int> predictions = predict(X);
    
    // Calculate accuracy
    size_t correct = 0;
    for (size_t i = 0; i < predictions.size() && i < y.size(); ++i) {
        if (predictions[i] == y[i]) {
            correct++;
        }
    }
    
    return static_cast<double>(correct) / y.size();
}

double SGDClassifier::predict_proba(const std::vector<double>& x) {
    // Calculate decision value
    double decision = 0.0;
    
    for (size_t i = 0; i < weights.size() && i < x.size(); ++i) {
        decision += weights[i] * x[i];
    }
    decision += intercept;
    
    // Convert to probability with sigmoid function
    return 1.0 / (1.0 + std::exp(-decision));
}

} // namespace nlp
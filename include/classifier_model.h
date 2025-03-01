/**
 * @file classifier_model.h
 * @brief Defines the ClassifierModel interface for sentiment classification.
 */

#pragma once

#include <vector>

namespace nlp {

/**
 * @class ClassifierModel
 * @brief Abstract base class for classification models.
 * 
 * This interface defines the methods that all classifier models must implement,
 * enabling polymorphic usage of different classification algorithms.
 */
class ClassifierModel {
public:
    /**
     * @brief Virtual destructor for proper cleanup in derived classes.
     */
    virtual ~ClassifierModel() = default;

    /**
     * @brief Trains the model on feature vectors and corresponding labels.
     * @param X Feature matrix where each row is a sample and each column a feature.
     * @param y Target labels corresponding to each sample.
     */
    virtual void fit(const std::vector<std::vector<double>>& X, 
                    const std::vector<int>& y) = 0;

    /**
     * @brief Predicts class labels for given samples.
     * @param X Feature matrix where each row is a sample to predict.
     * @return Vector of predicted class labels.
     */
    virtual std::vector<int> predict(const std::vector<std::vector<double>>& X) const = 0;

    /**
     * @brief Computes the decision function scores.
     * @param X Feature matrix.
     * @return Raw prediction scores for each sample.
     */
    virtual std::vector<double> decisionFunction(const std::vector<std::vector<double>>& X) const = 0;

    /**
     * @brief Evaluates model accuracy on a test set.
     * @param X Feature matrix of test samples.
     * @param y True labels for test samples.
     * @return Accuracy score between 0.0 and 1.0.
     */
    virtual double score(const std::vector<std::vector<double>>& X, 
                        const std::vector<int>& y) const = 0;
                        
    /**
     * @brief Computes probability estimates for samples.
     * @param X Feature matrix where each row is a sample.
     * @return Probability estimates for each sample.
     */
    virtual std::vector<double> predict_proba(const std::vector<std::vector<double>>& X) const = 0;
};

} // namespace nlp

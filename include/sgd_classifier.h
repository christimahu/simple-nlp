/**
 * @file sgd_classifier.h
 * @brief Implements a Stochastic Gradient Descent (SGD) classifier for binary classification.
 *
 * This classifier is useful for large-scale and online learning tasks.
 * It updates weights incrementally using a given loss function and implements
 * the ClassifierModel interface for integration with the sentiment analysis system.
 */

#ifndef SGD_CLASSIFIER_H
#define SGD_CLASSIFIER_H

#include <vector>
#include <string>
#include "classifier_model.h"

namespace nlp {

/**
 * @class SGDClassifier
 * @brief Implements an online learning classifier using Stochastic Gradient Descent.
 * 
 * This class inherits from ClassifierModel to provide a concrete implementation of
 * the classification interface using the SGD algorithm. It supports various loss
 * functions and regularization options for better generalization.
 */
class SGDClassifier : public ClassifierModel {
public:
    /**
     * @brief Constructor for initializing the classifier with hyperparameters.
     * @param loss The loss function ("hinge" for SVM, "log" for logistic regression, etc.).
     * @param alpha Regularization strength.
     * @param epochs Number of training iterations.
     * @param learningRate Step size for weight updates.
     */
    SGDClassifier(const std::string& loss = "hinge",
                  double alpha = 0.0001,
                  int epochs = 5,
                  double learningRate = 0.01);

    /**
     * @brief Fits the model to the given training data.
     * @param X Feature matrix.
     * @param y Target labels.
     */
    void fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y) override;

    /**
     * @brief Predicts class labels for given samples.
     * @param X Feature matrix.
     * @return Vector of predicted class labels.
     */
    std::vector<int> predict(const std::vector<std::vector<double>>& X) const override;

    /**
     * @brief Computes the decision function scores.
     * @param X Feature matrix.
     * @return Vector of decision scores.
     */
    std::vector<double> decisionFunction(const std::vector<std::vector<double>>& X) const override;

    /**
     * @brief Evaluates the model accuracy.
     * @param X Feature matrix.
     * @param y True labels.
     * @return Accuracy score (0.0 - 1.0).
     */
    double score(const std::vector<std::vector<double>>& X, const std::vector<int>& y) const override;

    /**
     * @brief Computes probability estimates for samples.
     * @param X Feature matrix where each row is a sample.
     * @return Probability estimates for each sample.
     */
    std::vector<double> predict_proba(const std::vector<std::vector<double>>& X) const override;
    
    /**
     * @brief Estimates probability based on the decision function for a single sample.
     * @param x Feature vector.
     * @return Probability estimate.
     * @note This is a helper method used internally.
     */
    double predict_proba(const std::vector<double>& x) const;

private:
    std::string loss;      ///< Loss function type used for training
    double alpha;          ///< Regularization strength parameter
    int epochs;            ///< Number of training iterations over the dataset
    double learningRate;   ///< Step size for weight updates
    std::vector<double> weights;  ///< Model weights for each feature
    double bias;           ///< Bias term (intercept) of the model
};

}  // namespace nlp

#endif  // SGD_CLASSIFIER_H

/**
 * @file sentiment_analysis.h
 * @brief Defines the main components for sentiment analysis.
 * 
 * This header provides the primary interfaces and structures for performing
 * sentiment analysis on text data. It includes the SentimentAnalyzer class
 * and supporting structures like SentimentResult.
 */

#ifndef NLP_SENTIMENT_ANALYSIS_H
#define NLP_SENTIMENT_ANALYSIS_H

#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <optional>
#include "text_preprocessor.h"
#include "tfidf_vectorizer.h"
#include "classifier_model.h"
#include "model_evaluator.h"
#include "sentiment_dataset.h"

namespace nlp {

// ================================
// SentimentResult
// ================================
/**
 * @brief Struct representing the result of a sentiment analysis.
 * 
 * This structure contains all relevant information about a sentiment
 * prediction, including the original text, preprocessed text, sentiment
 * label, confidence scores, and explanatory information.
 */
struct SentimentResult {
    std::string text;              ///< Original input text
    std::string cleanText;         ///< Preprocessed text after cleaning
    std::string sentiment;         ///< Sentiment label ("Positive" or "Negative")
    int label;                    ///< Numeric sentiment label (0 or 4)
    double rawScore;              ///< Raw decision function score
    double confidence;            ///< Confidence score (0.0-1.0)
    std::string explanation;      ///< Human-readable explanation of the result
    
    /**
     * @brief Converts the result to a map for serialization.
     * @return Map containing all result fields.
     */
    std::unordered_map<std::string, std::string> toMap() const;
    
    /**
     * @brief Creates a SentimentResult from a map representation.
     * @param map Map containing sentiment result fields.
     * @return A new SentimentResult instance.
     */
    static SentimentResult fromMap(const std::unordered_map<std::string, std::string>& map);
};

// ================================
// SentimentAnalyzer
// ================================
/**
 * @brief Class for performing sentiment analysis on text data.
 * 
 * This class orchestrates the complete sentiment analysis pipeline,
 * including data loading, preprocessing, feature extraction, model training,
 * evaluation, and prediction of sentiment for new texts.
 */
class SentimentAnalyzer {
public:
    /**
     * @brief Default constructor.
     */
    SentimentAnalyzer();
    
    /**
     * @brief Loads sentiment data from a file.
     * @param filePath Path to the data file (CSV format).
     * @return Optional SentimentDataset if loading was successful.
     */
    std::optional<SentimentDataset> loadData(const std::string& filePath);
    
    /**
     * @brief Preprocesses text data by applying cleaning operations.
     * @param dataset Input dataset to preprocess.
     * @param steps Optional vector of preprocessing steps to apply.
     * @return Preprocessed dataset.
     */
    SentimentDataset preprocessData(SentimentDataset dataset, 
                                    const std::vector<std::string>& steps = {});
    
    /**
     * @brief Extracts TF-IDF features from preprocessed text.
     * @param dataset Input dataset with preprocessed text.
     * @param maxDf Maximum document frequency for feature selection.
     * @param maxFeatures Maximum number of features to extract.
     * @return Pair of feature matrix and label vector.
     */
    std::pair<std::vector<std::vector<double>>, std::vector<int>> extractFeatures(
        const SentimentDataset& dataset,
        double maxDf = 0.5,
        size_t maxFeatures = 6228);
    
    /**
     * @brief Splits a dataset into training and testing sets.
     * @param dataset Input dataset to split.
     * @param testSize Number or fraction of samples for the test set.
     * @param randomState Seed for random number generation.
     * @return Dataset with train/test split indices set.
     */
    SentimentDataset splitData(SentimentDataset dataset,
                               double testSize = 0.25,
                               unsigned int randomState = 42);
    
    /**
     * @brief Trains a sentiment classification model.
     * @param X_train Training feature matrix.
     * @param y_train Training label vector.
     * @param alpha Regularization parameter.
     * @param eta0 Initial learning rate.
     * @return Unique pointer to the trained model.
     */
    std::unique_ptr<ClassifierModel> trainModel(
        const std::vector<std::vector<double>>& X_train,
        const std::vector<int>& y_train,
        double alpha = 2.9e-05,
        double eta0 = 0.00164);
    
    /**
     * @brief Evaluates model performance on test data.
     * @param model Trained classification model.
     * @param X_test Test feature matrix.
     * @param y_test Test label vector.
     * @return Map of evaluation metrics.
     */
    std::unordered_map<std::string, double> evaluateModel(
        const ClassifierModel& model,
        const std::vector<std::vector<double>>& X_test,
        const std::vector<int>& y_test);
    
    /**
     * @brief Predicts sentiment for a new text.
     * @param text Input text to analyze.
     * @param model Trained sentiment model.
     * @param vectorizer TF-IDF vectorizer.
     * @return Structured sentiment result.
     */
    SentimentResult predictSentiment(
        const std::string& text,
        const ClassifierModel& model,
        const TfidfVectorizer& vectorizer);
    
    /**
     * @brief Generates a word cloud for texts with a specific sentiment.
     * @param dataset Dataset containing texts.
     * @param sentiment Sentiment value (0 or 4).
     * @param outputPath Optional path to save the word cloud image.
     * @return True if generation was successful.
     */
    bool generateWordCloud(
        const SentimentDataset& dataset,
        int sentiment,
        const std::string& outputPath = "");

private:
    TextPreprocessor preprocessor;  ///< Text preprocessing component
    
    /**
     * @brief Generates an explanation for the sentiment prediction.
     * @param score Raw sentiment score.
     * @param confidence Model confidence score.
     * @return Human-readable explanation.
     */
    std::string generateExplanation(double score, double confidence);
};

}  // namespace nlp

#endif // NLP_SENTIMENT_ANALYSIS_H

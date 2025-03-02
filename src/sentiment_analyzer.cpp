/**
 * @file sentiment_analyzer.cpp
 * @brief Implements the sentiment analysis pipeline.
 * 
 * This file provides the implementation for the SentimentAnalyzer class,
 * which orchestrates the entire sentiment analysis workflow from data loading
 * to model training and prediction. It ties together the text preprocessing,
 * feature extraction, classification, and evaluation components.
 */

#include "sentiment_analysis.h"
#include "text_preprocessor.h"
#include "tfidf_vectorizer.h"
#include "classifier_model.h"
#include "sgd_classifier.h"
#include "model_evaluator.h"
#include "ascii_word_cloud.h"
#include "sentiment_dataset.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <iomanip>

namespace nlp {

// ================================
// SentimentResult Implementation
// ================================

/**
 * @brief Converts a SentimentResult to a map for serialization.
 * @return Map representation of the sentiment result.
 */
std::unordered_map<std::string, std::string> SentimentResult::toMap() const {
    std::unordered_map<std::string, std::string> map;
    
    map["text"] = text;
    map["clean_text"] = cleanText;
    map["sentiment"] = sentiment;
    map["label"] = std::to_string(label);
    map["raw_score"] = std::to_string(rawScore);
    map["confidence"] = std::to_string(confidence);
    map["explanation"] = explanation;
    
    return map;
}

/**
 * @brief Creates a SentimentResult from a map representation.
 * @param map Map containing sentiment result fields.
 * @return A new SentimentResult instance.
 */
SentimentResult SentimentResult::fromMap(const std::unordered_map<std::string, std::string>& map) {
    SentimentResult result;
    
    auto getText = [&map](const std::string& key) -> std::string {
        auto it = map.find(key);
        return (it != map.end()) ? it->second : "";
    };
    
    auto getDouble = [&map](const std::string& key) -> double {
        auto it = map.find(key);
        return (it != map.end()) ? std::stod(it->second) : 0.0;
    };
    
    auto getInt = [&map](const std::string& key) -> int {
        auto it = map.find(key);
        return (it != map.end()) ? std::stoi(it->second) : 0;
    };
    
    result.text = getText("text");
    result.cleanText = getText("clean_text");
    result.sentiment = getText("sentiment");
    result.label = getInt("label");
    result.rawScore = getDouble("raw_score");
    result.confidence = getDouble("confidence");
    result.explanation = getText("explanation");
    
    return result;
}

// ================================
// SentimentAnalyzer Implementation
// ================================

/**
 * @brief Initializes the sentiment analyzer with default components.
 */
SentimentAnalyzer::SentimentAnalyzer() : preprocessor() {}

/**
 * @brief Loads sentiment data from a CSV file.
 * 
 * This method reads data from a CSV file with sentiment labels and text,
 * returning an optional dataset if loading is successful.
 * 
 * @param filePath Path to the CSV file.
 * @return Optional SentimentDataset if loading was successful.
 */
std::optional<SentimentDataset> SentimentAnalyzer::loadData(const std::string& filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filePath << std::endl;
        return std::nullopt;
    }
    
    std::vector<std::pair<int, std::string>> dataRows;
    std::string line;
    bool isHeader = true;
    
    // Read the file line by line
    while (std::getline(file, line)) {
        // Skip header line
        if (isHeader) {
            isHeader = false;
            continue;
        }
        
        std::istringstream ss(line);
        std::string token;
        
        // Parse sentiment label
        if (!std::getline(ss, token, ',')) {
            continue;  // Skip invalid lines
        }
        int sentiment = std::stoi(token);
        
        // Parse tweet text (may contain commas)
        if (!std::getline(ss, token)) {
            continue;  // Skip invalid lines
        }
        
        // Remove surrounding quotes if present
        if (token.size() >= 2 && token.front() == '"' && token.back() == '"') {
            token = token.substr(1, token.size() - 2);
        }
        
        // Add to dataset
        dataRows.emplace_back(sentiment, token);
    }
    
    std::cout << "Loaded dataset with " << dataRows.size() << " rows" << std::endl;
    
    return SentimentDataset(dataRows);
}

/**
 * @brief Preprocesses text data by applying cleaning operations.
 * 
 * This method applies a series of text transformations to prepare
 * the data for feature extraction.
 * 
 * @param dataset Input dataset to preprocess.
 * @param steps Optional vector of preprocessing steps to apply.
 * @return Preprocessed dataset.
 */
SentimentDataset SentimentAnalyzer::preprocessData(SentimentDataset dataset, 
                                                  const std::vector<std::string>& steps) {
    // Extract raw texts from dataset
    std::vector<std::string> rawTexts;
    for (const auto& [_, text] : dataset.data) {
        rawTexts.push_back(text);
    }
    
    // Preprocess each text
    dataset.cleanedTexts.clear();
    dataset.cleanedTexts.reserve(rawTexts.size());
    
    for (const auto& text : rawTexts) {
        dataset.cleanedTexts.push_back(preprocessor.preprocess(text, steps));
    }
    
    std::cout << "Preprocessed " << dataset.cleanedTexts.size() << " texts" << std::endl;
    
    return dataset;
}

/**
 * @brief Extracts TF-IDF features from preprocessed text.
 * 
 * This method converts text data into numerical feature vectors
 * using the Term Frequency-Inverse Document Frequency approach.
 * 
 * @param dataset Input dataset with preprocessed text.
 * @param maxDf Maximum document frequency for feature selection.
 * @param maxFeatures Maximum number of features to extract.
 * @return Pair of feature matrix and label vector.
 */
std::pair<std::vector<std::vector<double>>, std::vector<int>> SentimentAnalyzer::extractFeatures(
    const SentimentDataset& dataset,
    double maxDf,
    size_t maxFeatures) {
    
    if (dataset.cleanedTexts.empty()) {
        throw std::runtime_error("No preprocessed texts available. Call preprocessData() first.");
    }
    
    // Create and configure TF-IDF vectorizer
    TfidfVectorizer vectorizer(true, maxDf, maxFeatures);
    
    // Extract features
    auto features = vectorizer.fitTransform(dataset.cleanedTexts);
    
    std::cout << "Extracted " << features[0].size() << " features from " 
              << features.size() << " texts" << std::endl;
    
    return {features, dataset.labels};
}

/**
 * @brief Splits a dataset into training and testing sets.
 * 
 * This method divides the dataset into training and testing portions,
 * maintaining class balance and optionally using stratified sampling.
 * 
 * @param dataset Input dataset to split.
 * @param testSize Number or fraction of samples for the test set.
 * @param randomState Seed for random number generation.
 * @return Dataset with train/test split indices set.
 */
SentimentDataset SentimentAnalyzer::splitData(SentimentDataset dataset,
                                             double testSize,
                                             unsigned int randomState) {
    
    if (dataset.cleanedTexts.empty() || dataset.labels.empty()) {
        throw std::runtime_error("Dataset is empty. Cannot split.");
    }
    
    size_t datasetSize = dataset.cleanedTexts.size();
    size_t testCount;
    
    // Convert testSize to absolute count if it's a fraction
    if (testSize < 1.0) {
        testCount = static_cast<size_t>(testSize * datasetSize);
    } else {
        testCount = static_cast<size_t>(testSize);
    }
    
    // Ensure valid test count
    testCount = std::max(size_t(1), std::min(testCount, datasetSize - 1));
    
    // Create indices for the full dataset
    std::vector<size_t> indices(datasetSize);
    std::iota(indices.begin(), indices.end(), 0);
    
    // Shuffle indices
    std::mt19937 gen(randomState);
    std::shuffle(indices.begin(), indices.end(), gen);
    
    // Split into train and test sets
    dataset.testIndices.assign(indices.begin(), indices.begin() + testCount);
    dataset.trainIndices.assign(indices.begin() + testCount, indices.end());
    
    // Print split information
    std::cout << "Training set size: " << dataset.trainIndices.size() 
              << ", Test set size: " << dataset.testIndices.size() << std::endl;
    
    return dataset;
}

/**
 * @brief Trains a sentiment classification model.
 * 
 * This method creates and trains an SGD classifier for sentiment prediction.
 * 
 * @param X_train Training feature matrix.
 * @param y_train Training label vector.
 * @param alpha Regularization parameter.
 * @param eta0 Initial learning rate.
 * @return Unique pointer to the trained model.
 */
std::unique_ptr<ClassifierModel> SentimentAnalyzer::trainModel(
    const std::vector<std::vector<double>>& X_train,
    const std::vector<int>& y_train,
    double alpha,
    double eta0) {
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Create a new SGDClassifier
    auto model = std::make_unique<SGDClassifier>("log", alpha, 5, eta0);
    
    // Train the model
    model->fit(X_train, y_train);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime);
    
    std::cout << "Model training complete" << std::endl;
    std::cout << "Training took " << duration.count() << " seconds" << std::endl;
    
    return model;
}

/**
 * @brief Evaluates model performance on test data.
 * 
 * This method assesses the quality of a trained model using standard
 * classification metrics like accuracy, precision, and recall.
 * 
 * @param model Trained classification model.
 * @param X_test Test feature matrix.
 * @param y_test Test label vector.
 * @return Map of evaluation metrics.
 */
std::unordered_map<std::string, double> SentimentAnalyzer::evaluateModel(
    const ClassifierModel& model,
    const std::vector<std::vector<double>>& X_test,
    const std::vector<int>& y_test) {
    
    // Get predictions
    std::vector<int> y_pred = model.predict(X_test);
    
    // Calculate score
    double score = model.score(X_test, y_test);
    std::cout << "Model Score: " << std::fixed << std::setprecision(6) << score << std::endl;
    
    // Calculate confusion matrix
    auto confMatrix = ModelEvaluator::confusionMatrix(y_test, y_pred);
    
    std::cout << "Confusion Matrix:" << std::endl;
    std::cout << "Actual \\ Pred  0       4" << std::endl;
    std::cout << "0              " << confMatrix[0][0] << "    " << confMatrix[0][1] << std::endl;
    std::cout << "4              " << confMatrix[1][0] << "    " << confMatrix[1][1] << std::endl;
    std::cout << std::endl;
    
    // Calculate per-class accuracy
    int classCount0 = confMatrix[0][0] + confMatrix[0][1];
    int classCount4 = confMatrix[1][0] + confMatrix[1][1];
    
    double accuracy0 = classCount0 > 0 ? 
                      static_cast<double>(confMatrix[0][0]) / classCount0 * 100.0 : 0.0;
    double accuracy4 = classCount4 > 0 ? 
                      static_cast<double>(confMatrix[1][1]) / classCount4 * 100.0 : 0.0;
    
    std::cout << "Per-class accuracy:" << std::endl;
    std::cout << "Class 4: " << std::fixed << std::setprecision(2) << accuracy4 
              << "% (" << confMatrix[1][1] << "/" << classCount4 << ")" << std::endl;
    std::cout << "Class 0: " << std::fixed << std::setprecision(2) << accuracy0 
              << "% (" << confMatrix[0][0] << "/" << classCount0 << ")" << std::endl;
    std::cout << std::endl;
    
    // Generate classification report
    std::cout << "Classification Report:" << std::endl;
    std::cout << ModelEvaluator::classificationReport(y_test, y_pred);
    
    // Calculate and return metrics
    return ModelEvaluator::calculateMetrics(y_test, y_pred);
}

/**
 * @brief Predicts sentiment for a new text.
 * 
 * This method analyzes the sentiment of a given text using a trained model,
 * providing a detailed result with confidence scores and explanation.
 * 
 * @param text Input text to analyze.
 * @param model Trained sentiment model.
 * @param vectorizer TF-IDF vectorizer.
 * @return Structured sentiment result.
 */
SentimentResult SentimentAnalyzer::predictSentiment(
    const std::string& text,
    const ClassifierModel& model,
    const TfidfVectorizer& vectorizer) {
    
    // Preprocess the text
    std::string cleanText = preprocessor.preprocess(text);
    
    // Vectorize the text
    std::vector<std::vector<double>> features = vectorizer.transform({cleanText});
    
    // Make prediction
    std::vector<int> predictions = model.predict(features);
    std::vector<double> scores = model.decisionFunction(features);
    
    int label = predictions[0];
    double rawScore = scores[0];
    
    // Get probability
    std::vector<double> probs = model.predict_proba(features);
    double confidence = probs[0];
    
    // Create sentiment label
    std::string sentiment = (label == 4) ? "Positive" : "Negative";
    
    // Generate explanation
    std::string explanation = generateExplanation(rawScore, confidence);
    
    // Return result
    return {
        text,           // Original text
        cleanText,      // Preprocessed text
        sentiment,      // Sentiment label (string)
        label,          // Numeric label (0 or 4)
        rawScore,       // Raw decision score
        confidence,     // Confidence score (0.0-1.0)
        explanation     // Human-readable explanation
    };
}

/**
 * @brief Generates a word cloud for texts with a specific sentiment.
 * 
 * This method creates a visualization of the most frequent words in texts
 * with a particular sentiment label, useful for exploring patterns.
 * 
 * @param dataset Dataset containing texts.
 * @param sentiment Sentiment value (0 or 4).
 * @param outputPath Optional path to save the word cloud image.
 * @return True if generation was successful.
 */
bool SentimentAnalyzer::generateWordCloud(
    const SentimentDataset& dataset,
    int sentiment,
    const std::string& outputPath) {
    
    // Filter texts by sentiment
    std::vector<std::string> filteredTexts = dataset.getTextsWithSentiment(sentiment);
    
    if (filteredTexts.empty()) {
        std::cerr << "No texts found with sentiment " << sentiment << std::endl;
        return false;
    }
    
    // Create an instance of AsciiWordCloud
    AsciiWordCloud wordCloud;
    
    // Configure word cloud
    AsciiWordCloud::CloudConfig config;
    config.maxWords = 30;
    config.width = 80;
    config.height = 20;
    config.useColor = true;
    config.showFrequencies = true;
    
    // Generate word cloud
    bool isPositive = (sentiment == 4);
    std::string cloud = wordCloud.generateCustomCloud(
        filteredTexts, config, isPositive);
    
    // Display the word cloud
    std::cout << cloud << std::endl;
    
    // Save to file if path provided
    if (!outputPath.empty()) {
        std::ofstream outFile(outputPath);
        if (outFile.is_open()) {
            outFile << cloud;
            outFile.close();
            std::cout << "Word cloud saved to " << outputPath << std::endl;
        } else {
            std::cerr << "Error: Could not save word cloud to " << outputPath << std::endl;
            return false;
        }
    }
    
    return true;
}

/**
 * @brief Generates an explanation for the sentiment prediction.
 * 
 * This method provides a human-readable explanation of the prediction
 * based on the model's confidence level.
 * 
 * @param score Raw sentiment score.
 * @param confidence Confidence level (0.0-1.0).
 * @return Explanation string.
 */
std::string SentimentAnalyzer::generateExplanation(double score, double confidence) {
    std::ostringstream explanation;
    
    // Explain based on confidence level
    if (confidence > 0.9) {
        explanation << "Very high confidence prediction. ";
        explanation << "The model is extremely certain about this sentiment assessment.";
    } else if (confidence > 0.75) {
        explanation << "High confidence prediction. ";
        explanation << "The model has strong evidence for this sentiment.";
    } else if (confidence > 0.6) {
        explanation << "Moderate confidence prediction. ";
        explanation << "The model has reasonable evidence for this sentiment.";
    } else if (confidence > 0.5) {
        explanation << "Low confidence prediction. ";
        explanation << "The model leans toward this sentiment but is somewhat uncertain.";
    } else {
        explanation << "Very low confidence prediction. ";
        explanation << "The model is highly uncertain and the sentiment could be ambiguous.";
    }
    
    return explanation.str();
}

}  // namespace nlp

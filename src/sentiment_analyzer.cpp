/**
 * @file sentiment_analyzer.cpp
 * @brief Implementation of the SentimentAnalyzer class
 * 
 * This file contains the implementation of the SentimentAnalyzer class,
 * which orchestrates the entire sentiment analysis process.
 */

#include "sentiment_analysis.h"
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <iostream>
#include <set>
#include <iomanip>
#include <functional>
#include <numeric>

namespace nlp {

/**
 * Constructor initializes the analyzer with default settings.
 */
SentimentAnalyzer::SentimentAnalyzer() 
    : preprocessor() {
}

/**
 * Loads tweet dataset from a CSV file.
 * 
 * @param dataPath Path to the CSV file
 * @return Optional containing dataset (empty if loading failed)
 */
std::optional<SentimentDataset> SentimentAnalyzer::loadData(const std::string& dataPath) {
    std::ifstream file(dataPath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << dataPath << std::endl;
        return std::nullopt;
    }
    
    SentimentDataset dataset;
    
    // Skip header if present
    std::string headerLine;
    std::getline(file, headerLine);
    
    // Process each line
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string sentimentStr, tweetText;
        
        // Assume CSV format: sentiment_label,tweet_text
        if (std::getline(ss, sentimentStr, ',') && std::getline(ss, tweetText)) {
            try {
                int sentiment = std::stoi(sentimentStr);
                dataset.data.push_back({sentiment, tweetText});
            } catch (const std::exception& e) {
                std::cerr << "Error parsing line: " << line << std::endl;
                std::cerr << e.what() << std::endl;
            }
        }
    }
    
    std::cout << "Loaded dataset with " << dataset.data.size() << " rows" << std::endl;
    
    // Show a few samples
    if (!dataset.data.empty()) {
        std::cout << "Sample data:" << std::endl;
        for (size_t i = 0; i < std::min(dataset.data.size(), size_t(5)); ++i) {
            std::cout << "Sentiment: " << dataset.data[i].first << ", Text: " << dataset.data[i].second << std::endl;
        }
    }
    
    if (dataset.data.empty()) {
        return std::nullopt;
    }
    return dataset;
}

/**
 * Preprocesses the tweets in the dataset.
 * 
 * @param dataset The dataset to preprocess (moved)
 * @param steps List of preprocessing steps to apply
 * @return Preprocessed dataset
 */
SentimentDataset SentimentAnalyzer::preprocessData(
    SentimentDataset&& dataset, 
    const std::vector<std::string>& steps) {
    
    if (dataset.data.empty()) {
        std::cerr << "Error: No data loaded." << std::endl;
        return std::move(dataset);
    }
    
    dataset.cleanedTexts.clear();
    dataset.labels.clear();
    
    // Process each tweet
    for (const auto& [sentiment, text] : dataset.data) {
        std::string cleanText = preprocessor.preprocess(text, steps);
        dataset.cleanedTexts.push_back(cleanText);
        dataset.labels.push_back(sentiment);
    }
    
    std::cout << "Preprocessing complete. Sample of preprocessed tweets:" << std::endl;
    
    // Show sample results
    for (size_t i = 0; i < std::min(dataset.cleanedTexts.size(), size_t(5)); ++i) {
        std::cout << "Original: " << dataset.data[i].second << std::endl;
        std::cout << "Cleaned: " << dataset.cleanedTexts[i] << std::endl << std::endl;
    }
    
    return std::move(dataset);
}

/**
 * Generates a word cloud for tweets with a specific sentiment.
 * 
 * @param dataset The sentiment dataset
 * @param sentimentValue The sentiment label value (0 for negative, 4 for positive)
 * @param outputPath Path to save the word cloud text (optional)
 * @return True if word cloud generation was successful
 */
bool SentimentAnalyzer::generateWordCloud(
    const SentimentDataset& dataset, 
    int sentimentValue, 
    const std::string& outputPath) {
    
    if (dataset.data.empty() || dataset.cleanedTexts.empty()) {
        std::cerr << "Error: No data or preprocessed data available" << std::endl;
        return false;
    }
    
    // Collect texts with the specified sentiment
    std::vector<std::string> sentimentTexts;
    
    for (size_t i = 0; i < dataset.data.size(); ++i) {
        if (dataset.data[i].first == sentimentValue) {
            sentimentTexts.push_back(dataset.cleanedTexts[i]);
        }
    }
    
    if (sentimentTexts.empty()) {
        std::cerr << "Error: No texts found with sentiment " << sentimentValue << std::endl;
        return false;
    }
    
    std::cout << "Generating word cloud for sentiment " << sentimentValue 
              << " using " << sentimentTexts.size() << " texts..." << std::endl;
    
    bool isPositive = (sentimentValue == 4);
    
    // Generate and display word cloud
    AsciiWordCloud::displayWordCloud(sentimentTexts, 30, 100, 20, isPositive);
    
    // If output path is provided, save to file
    if (!outputPath.empty()) {
        std::ofstream outFile(outputPath);
        if (outFile.is_open()) {
            outFile << AsciiWordCloud::generateWordCloud(sentimentTexts, 30, 100, 20, isPositive);
            outFile.close();
            std::cout << "Word cloud saved to " << outputPath << std::endl;
            return true;
        } else {
            std::cerr << "Error: Could not open file " << outputPath << " for writing" << std::endl;
            return false;
        }
    }
    
    return true;
}

/**
 * Extracts TF-IDF features from preprocessed tweets.
 * 
 * @param dataset The preprocessed dataset
 * @param maxDf Maximum document frequency for terms
 * @param maxFeatures Maximum number of features to extract
 * @return Pair of (features, labels)
 */
std::pair<std::vector<std::vector<double>>, std::vector<int>> 
SentimentAnalyzer::extractFeatures(
    const SentimentDataset& dataset, 
    double maxDf, 
    size_t maxFeatures) {
    
    if (dataset.cleanedTexts.empty()) {
        std::cerr << "Error: No preprocessed data available" << std::endl;
        return {{}, {}};
    }
    
    // Create TF-IDF vectorizer with specified parameters
    TfidfVectorizer vectorizer(true, maxDf, maxFeatures);
    
    // Extract features using the vectorizer
    auto features = vectorizer.fitTransform(dataset.cleanedTexts);
    
    std::cout << "Extracted " << features[0].size() << " features from " 
              << features.size() << " tweets" << std::endl;
    
    return {features, dataset.labels};
}

/**
 * Splits the dataset into training and testing sets.
 * 
 * @param dataset The dataset to split (moved)
 * @param testSize Size of the test set
 * @param randomState Random seed for reproducibility
 * @return Dataset with train/test split indices
 */
SentimentDataset SentimentAnalyzer::splitData(
    SentimentDataset&& dataset, 
    size_t testSize, 
    unsigned int randomState) {
    
    if (dataset.cleanedTexts.empty() || dataset.labels.empty()) {
        std::cerr << "Error: No features or labels available" << std::endl;
        return std::move(dataset);
    }
    
    // Create indices
    std::vector<size_t> indices(dataset.cleanedTexts.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    // Shuffle indices
    std::mt19937 g(randomState);
    std::shuffle(indices.begin(), indices.end(), g);
    
    // Determine test size
    if (testSize > dataset.cleanedTexts.size()) {
        testSize = dataset.cleanedTexts.size() / 4; // Default to 25% if too large
    }
    size_t trainSize = dataset.cleanedTexts.size() - testSize;
    
    // Split the indices
    dataset.trainIndices.assign(indices.begin(), indices.begin() + trainSize);
    dataset.testIndices.assign(indices.begin() + trainSize, indices.end());
    
    // Count class distribution
    std::unordered_map<int, int> classCounts;
    for (int label : dataset.labels) {
        classCounts[label]++;
    }
    
    std::cout << "Overall class distribution:" << std::endl;
    for (const auto& [label, count] : classCounts) {
        std::cout << "Class " << label << ": " << count << " samples (" 
                 << (static_cast<double>(count) / dataset.labels.size() * 100.0) << "%)" << std::endl;
    }
    
    // Count training set distribution
    std::unordered_map<int, int> trainCounts;
    for (size_t idx : dataset.trainIndices) {
        trainCounts[dataset.labels[idx]]++;
    }
    
    std::cout << "Training set class distribution:" << std::endl;
    for (const auto& [label, count] : trainCounts) {
        std::cout << "Class " << label << ": " << count << " samples (" 
                 << (static_cast<double>(count) / dataset.trainIndices.size() * 100.0) << "%)" << std::endl;
    }
    
    // Count test set distribution
    std::unordered_map<int, int> testCounts;
    for (size_t idx : dataset.testIndices) {
        testCounts[dataset.labels[idx]]++;
    }
    
    std::cout << "Test set class distribution:" << std::endl;
    for (const auto& [label, count] : testCounts) {
        std::cout << "Class " << label << ": " << count << " samples (" 
                 << (static_cast<double>(count) / dataset.testIndices.size() * 100.0) << "%)" << std::endl;
    }
    
    std::cout << "Training set size: " << dataset.trainIndices.size() 
              << ", Test set size: " << dataset.testIndices.size() << std::endl;
    
    return std::move(dataset);
}

/**
 * Trains a sentiment classifier on training data.
 * 
 * @param X_train Training feature matrix
 * @param y_train Training labels
 * @param alpha Regularization parameter
 * @param eta0 Initial learning rate
 * @return Unique pointer to trained classifier model
 */
std::unique_ptr<ClassifierModel> SentimentAnalyzer::trainModel(
    const std::vector<std::vector<double>>& X_train,
    const std::vector<int>& y_train,
    double alpha, 
    double eta0) {
    
    if (X_train.empty() || y_train.empty()) {
        std::cerr << "Error: No training data available" << std::endl;
        return nullptr;
    }
    
    // Create and train the classifier
    std::unique_ptr<ClassifierModel> model = 
        std::make_unique<SGDClassifier>("modified_huber", "adaptive", "elasticnet", alpha, eta0);
    
    model->fit(X_train, y_train);
    
    std::cout << "Model training complete" << std::endl;
    
    return model;
}

/**
 * Evaluates the trained sentiment classifier.
 * 
 * @param model The trained model to evaluate
 * @param X_test Test feature matrix
 * @param y_test Test labels
 * @return Map of evaluation metrics
 */
std::unordered_map<std::string, double> SentimentAnalyzer::evaluateModel(
    const ClassifierModel& model,
    const std::vector<std::vector<double>>& X_test, 
    const std::vector<int>& y_test) {
    
    if (X_test.empty() || y_test.empty()) {
        std::cerr << "Error: No test data available" << std::endl;
        return {{"error", 1.0}};
    }
    
    // Get predictions
    std::vector<int> y_pred = model.predict(X_test);
    
    // Calculate accuracy
    double score = model.score(X_test, y_test);
    std::cout << "Model Score: " << std::fixed << std::setprecision(6) << score << std::endl;
    
    // Generate confusion matrix
    std::vector<std::vector<int>> confMatrix = ModelEvaluator::confusionMatrix(y_test, y_pred);
    
    // Find unique classes in the test set and predictions
    std::set<int> classes;
    for (int label : y_test) classes.insert(label);
    for (int label : y_pred) classes.insert(label);
    
    // Create a mapping from class value to index
    std::unordered_map<int, size_t> classIndices;
    size_t idx = 0;
    for (int classVal : classes) {
        classIndices[classVal] = idx++;
    }
    
    // Print confusion matrix with labels
    std::cout << "Confusion Matrix:" << std::endl;
    
    // Print header row with predicted class labels
    std::cout << std::setw(15) << "Actual \\ Pred";
    for (int classVal : classes) {
        std::cout << std::setw(8) << classVal;
    }
    std::cout << std::endl;
    
    // Print matrix rows
    for (int classVal : classes) {
        std::cout << std::setw(15) << classVal;
        
        size_t rowIdx = classIndices[classVal];
        for (size_t j = 0; j < confMatrix[rowIdx].size(); ++j) {
            std::cout << std::setw(8) << confMatrix[rowIdx][j];
        }
        std::cout << std::endl;
    }
    
    // Calculate per-class metrics
    std::unordered_map<int, int> classTotals;
    std::unordered_map<int, int> correctPredictions;
    
    for (size_t i = 0; i < y_test.size(); ++i) {
        classTotals[y_test[i]]++;
        if (y_test[i] == y_pred[i]) {
            correctPredictions[y_test[i]]++;
        }
    }
    
    // Print per-class accuracy
    std::cout << "\nPer-class accuracy:" << std::endl;
    for (const auto& [classVal, count] : classTotals) {
        double accuracy = static_cast<double>(correctPredictions[classVal]) / count;
        std::cout << "Class " << classVal << ": " << std::fixed << std::setprecision(2) 
                 << (accuracy * 100.0) << "% (" << correctPredictions[classVal] 
                 << "/" << count << ")" << std::endl;
    }
    
    // Generate detailed classification report
    std::vector<std::string> classNames;
    for (int classVal : classes) {
        classNames.push_back(classVal == 0 ? "Negative (0)" : "Positive (4)");
    }
    
    std::string classReport = ModelEvaluator::classificationReport(y_test, y_pred, classNames);
    std::cout << "\nClassification Report:" << std::endl;
    std::cout << classReport << std::endl;
    
    // Calculate prediction distribution
    std::unordered_map<int, int> predCounts;
    for (int pred : y_pred) {
        predCounts[pred]++;
    }
    
    std::cout << "\nPrediction distribution:" << std::endl;
    for (const auto& [classVal, count] : predCounts) {
        double percentage = static_cast<double>(count) / y_pred.size() * 100.0;
        std::cout << "Class " << classVal << ": " << count << " predictions (" 
                 << std::fixed << std::setprecision(2) << percentage << "%)" << std::endl;
    }
    
    // Calculate and return metrics
    return ModelEvaluator::calculateMetrics(y_test, y_pred);
}

/**
 * Predicts the sentiment of new text.
 * 
 * @param text The text to analyze
 * @param model The trained model
 * @param vectorizer The trained vectorizer
 * @return Sentiment prediction result
 */
SentimentResult SentimentAnalyzer::predictSentiment(
    const std::string& text,
    const ClassifierModel& model,
    const TfidfVectorizer& vectorizer) {
    
    // Preprocess the text
    std::string cleanText = preprocessor.preprocess(text);
    
    // Check if text was properly preprocessed
    if (cleanText.empty()) {
        std::cerr << "Warning: Text was completely filtered out during preprocessing" << std::endl;
        // Use original text as fallback
        cleanText = text;
    }
    
    // Extract features (need to use a vector to match the interface)
    std::vector<std::string> textVec = {cleanText};
    std::vector<std::vector<double>> features = vectorizer.transform(textVec);
    
    // Make prediction
    int sentiment = model.predict(features)[0];
    
    // Get confidence scores
    std::vector<double> decisionValues = model.decisionFunction(features);
    double rawScore = decisionValues[0];
    
    // Scale very large scores for better interpretation
    double scaledScore = rawScore;
    if (std::abs(rawScore) > 10.0) {
        scaledScore = 10.0 * (rawScore / std::abs(rawScore));
    }
    
    double confidence = std::abs(scaledScore);
    
    // Calculate probability estimate using sigmoid
    double probability = 1.0 / (1.0 + std::exp(-scaledScore));
    
    // Interpret the prediction
    std::string sentimentLabel = (sentiment == 4) ? "Positive" : "Negative";
    
    // Generate explanation
    std::string explanation = generateExplanation(rawScore, confidence);
    
    // Create and return sentiment result
    SentimentResult result;
    result.text = text;
    result.cleanText = cleanText;
    result.sentiment = sentimentLabel;
    result.rawScore = rawScore;
    result.scaledScore = scaledScore;
    result.confidence = confidence;
    result.probability = probability;
    result.explanation = explanation;
    
    return result;
}

/**
 * Generates a human-readable explanation of the sentiment prediction.
 * 
 * @param score Decision function score
 * @param confidence Confidence value
 * @return Explanation text
 */
std::string SentimentAnalyzer::generateExplanation(double score, double confidence) const {
    std::stringstream explanation;
    
    // Generate explanation based on confidence level
    if (confidence <= 2.0) {
        explanation << "The model slightly leans toward ";
    } else if (confidence <= 5.0) {
        explanation << "The model suggests that this text is ";
    } else if (confidence <= 8.0) {
        explanation << "The model is confident that this text is ";
    } else {
        explanation << "The model is very confident that this text is ";
    }
    
    explanation << ((score > 0) ? "positive" : "negative");
    explanation << " (score: " << std::fixed << std::setprecision(2) << score << ").";
    
    return explanation.str();
}

/**
 * Convert SentimentResult to a string map for serialization.
 * 
 * @return Map representation of the result
 */
std::map<std::string, std::string> SentimentResult::toMap() const {
    return {
        {"text", text},
        {"clean_text", cleanText},
        {"sentiment", sentiment},
        {"raw_score", std::to_string(rawScore)},
        {"scaled_score", std::to_string(scaledScore)},
        {"confidence", std::to_string(confidence)},
        {"probability", std::to_string(probability)},
        {"explanation", explanation}
    };
}

/**
 * Create a SentimentResult from a string map.
 * 
 * @param map Map representation of a result
 * @return Reconstructed SentimentResult
 */
SentimentResult SentimentResult::fromMap(const std::map<std::string, std::string>& map) {
    SentimentResult result;
    
    // Extract values with safety checks
    auto extractString = [&map](const std::string& key) -> std::string {
        auto it = map.find(key);
        return (it != map.end()) ? it->second : "";
    };
    
    auto extractDouble = [&map](const std::string& key) -> double {
        auto it = map.find(key);
        if (it != map.end()) {
            try {
                return std::stod(it->second);
            } catch (...) {
                return 0.0;
            }
        }
        return 0.0;
    };
    
    // Set fields from map
    result.text = extractString("text");
    result.cleanText = extractString("clean_text");
    result.sentiment = extractString("sentiment");
    result.rawScore = extractDouble("raw_score");
    result.scaledScore = extractDouble("scaled_score");
    result.confidence = extractDouble("confidence");
    result.probability = extractDouble("probability");
    result.explanation = extractString("explanation");
    
    return result;
}

} // namespace nlp

#include "sentiment_analysis.h"
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <iostream>
#include <set>
#include <iomanip>

namespace nlp {

SentimentAnalyzer::SentimentAnalyzer() 
    : preprocessor(), vectorizer(nullptr), model(nullptr) {
}

bool SentimentAnalyzer::loadData(const std::string& dataPath) {
    std::ifstream file(dataPath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << dataPath << std::endl;
        return false;
    }
    
    data.clear();
    std::string line;
    
    // Skip header if present
    std::getline(file, line);
    
    // Process each line
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string sentimentStr, tweetText;
        
        // Assume CSV format: sentiment_label,tweet_text
        if (std::getline(ss, sentimentStr, ',') && std::getline(ss, tweetText)) {
            try {
                int sentiment = std::stoi(sentimentStr);
                data.push_back({sentiment, tweetText});
            } catch (const std::exception& e) {
                std::cerr << "Error parsing line: " << line << std::endl;
                std::cerr << e.what() << std::endl;
            }
        }
    }
    
    std::cout << "Loaded dataset with " << data.size() << " rows" << std::endl;
    
    // Show a few samples
    if (!data.empty()) {
        std::cout << "Sample data:" << std::endl;
        for (size_t i = 0; i < std::min(data.size(), size_t(5)); ++i) {
            std::cout << "Sentiment: " << data[i].first << ", Text: " << data[i].second << std::endl;
        }
    }
    
    return !data.empty();
}

bool SentimentAnalyzer::preprocessData(const std::vector<std::string>& steps) {
    if (data.empty()) {
        std::cerr << "Error: No data loaded. Call loadData() first." << std::endl;
        return false;
    }
    
    cleanedTexts.clear();
    labels.clear();
    
    // Process each tweet
    for (const auto& [sentiment, text] : data) {
        std::string cleanText = preprocessor.preprocess(text, steps);
        cleanedTexts.push_back(cleanText);
        labels.push_back(sentiment);
    }
    
    std::cout << "Preprocessing complete. Sample of preprocessed tweets:" << std::endl;
    for (size_t i = 0; i < std::min(cleanedTexts.size(), size_t(5)); ++i) {
        std::cout << "Original: " << data[i].second << std::endl;
        std::cout << "Cleaned: " << cleanedTexts[i] << std::endl << std::endl;
    }
    
    return true;
}

bool SentimentAnalyzer::generateWordCloud(int sentimentValue, const std::string& outputPath) {
    if (data.empty() || cleanedTexts.empty()) {
        std::cerr << "Error: No data or preprocessed data available" << std::endl;
        return false;
    }
    
    // Collect texts with the specified sentiment
    std::vector<std::string> sentimentTexts;
    
    for (size_t i = 0; i < data.size(); ++i) {
        if (data[i].first == sentimentValue) {
            sentimentTexts.push_back(cleanedTexts[i]);
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
        } else {
            std::cerr << "Error: Could not open file " << outputPath << " for writing" << std::endl;
            return false;
        }
    }
    
    return true;
}

bool SentimentAnalyzer::extractFeatures(double maxDf, size_t maxFeatures) {
    if (cleanedTexts.empty()) {
        std::cerr << "Error: No preprocessed data available" << std::endl;
        return false;
    }
    
    // Create TF-IDF vectorizer
    vectorizer = std::make_unique<TfidfVectorizer>(true, maxDf, maxFeatures);
    
    // Extract features
    features = vectorizer->fitTransform(cleanedTexts);
    
    std::cout << "Extracted " << features[0].size() << " features from " 
              << features.size() << " tweets" << std::endl;
    
    return !features.empty();
}

bool SentimentAnalyzer::splitData(size_t testSize, unsigned int randomState) {
    if (features.empty() || labels.empty()) {
        std::cerr << "Error: No features or labels available" << std::endl;
        return false;
    }
    
    // Create indices
    std::vector<size_t> indices(features.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    // Shuffle indices
    std::mt19937 g(randomState);
    std::shuffle(indices.begin(), indices.end(), g);
    
    // Determine test size
    if (testSize > features.size()) {
        testSize = features.size() / 4; // Default to 25% if too large
    }
    size_t trainSize = features.size() - testSize;
    
    // Split the data
    xTrain.resize(trainSize);
    yTrain.resize(trainSize);
    xTest.resize(testSize);
    yTest.resize(testSize);
    
    // Count class distribution before splitting
    std::unordered_map<int, int> class_counts;
    for (int label : labels) {
        class_counts[label]++;
    }
    
    std::cout << "Overall class distribution:" << std::endl;
    for (const auto& [label, count] : class_counts) {
        std::cout << "Class " << label << ": " << count << " samples (" 
                 << (static_cast<double>(count) / labels.size() * 100.0) << "%)" << std::endl;
    }
    
    // Split the data while maintaining class distribution
    for (size_t i = 0; i < trainSize; ++i) {
        xTrain[i] = features[indices[i]];
        yTrain[i] = labels[indices[i]];
    }
    
    for (size_t i = 0; i < testSize; ++i) {
        xTest[i] = features[indices[i + trainSize]];
        yTest[i] = labels[indices[i + trainSize]];
    }
    
    // Count classes in training set
    class_counts.clear();
    for (int label : yTrain) {
        class_counts[label]++;
    }
    
    std::cout << "Training set class distribution:" << std::endl;
    for (const auto& [label, count] : class_counts) {
        std::cout << "Class " << label << ": " << count << " samples (" 
                 << (static_cast<double>(count) / yTrain.size() * 100.0) << "%)" << std::endl;
    }
    
    // Count classes in test set
    class_counts.clear();
    for (int label : yTest) {
        class_counts[label]++;
    }
    
    std::cout << "Test set class distribution:" << std::endl;
    for (const auto& [label, count] : class_counts) {
        std::cout << "Class " << label << ": " << count << " samples (" 
                 << (static_cast<double>(count) / yTest.size() * 100.0) << "%)" << std::endl;
    }
    
    std::cout << "Training set size: " << xTrain.size() 
              << ", Test set size: " << xTest.size() << std::endl;
    
    return true;
}

bool SentimentAnalyzer::trainModel(double alpha, double eta0) {
    if (xTrain.empty() || yTrain.empty()) {
        std::cerr << "Error: No training data available" << std::endl;
        return false;
    }
    
    // Create and train the classifier
    model = std::make_unique<SGDClassifier>("modified_huber", "adaptive", "elasticnet", alpha, eta0);
    model->fit(xTrain, yTrain);
    
    std::cout << "Model training complete" << std::endl;
    
    return true;
}

bool SentimentAnalyzer::evaluateModel() {
    if (!model || xTest.empty() || yTest.empty()) {
        std::cerr << "Error: No trained model or test data available" << std::endl;
        return false;
    }
    
    // Get predictions
    std::vector<int> yPred = model->predict(xTest);
    
    // Calculate accuracy
    double score = model->score(xTest, yTest);
    std::cout << "Model Score: " << std::fixed << std::setprecision(6) << score << std::endl;
    
    // Generate confusion matrix
    std::vector<std::vector<int>> confMatrix = ModelEvaluator::confusionMatrix(yTest, yPred);
    
    // Find unique classes in the test set and predictions
    std::set<int> classes;
    for (int label : yTest) classes.insert(label);
    for (int label : yPred) classes.insert(label);
    
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
    
    // Print each row with actual class label
    for (int classVal : classes) {
        std::cout << std::setw(15) << classVal;
        size_t rowIdx = classIndices[classVal];
        for (size_t j = 0; j < confMatrix[rowIdx].size(); ++j) {
            std::cout << std::setw(8) << confMatrix[rowIdx][j];
        }
        std::cout << std::endl;
    }
    
    // Calculate per-class metrics
    int total = 0;
    std::unordered_map<int, int> classTotals;
    std::unordered_map<int, int> correctPredictions;
    
    for (size_t i = 0; i < yTest.size(); ++i) {
        classTotals[yTest[i]]++;
        total++;
        if (yTest[i] == yPred[i]) {
            correctPredictions[yTest[i]]++;
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
    
    std::string classReport = ModelEvaluator::classificationReport(yTest, yPred, classNames);
    std::cout << "\nClassification Report:" << std::endl;
    std::cout << classReport << std::endl;
    
    // Calculate prediction distribution
    std::unordered_map<int, int> predCounts;
    for (int pred : yPred) {
        predCounts[pred]++;
    }
    
    std::cout << "\nPrediction distribution:" << std::endl;
    for (const auto& [classVal, count] : predCounts) {
        double percentage = static_cast<double>(count) / yPred.size() * 100.0;
        std::cout << "Class " << classVal << ": " << count << " predictions (" 
                 << std::fixed << std::setprecision(2) << percentage << "%)" << std::endl;
    }
    
    return true;
}

std::map<std::string, std::string> SentimentAnalyzer::predictSentiment(const std::string& text) {
    if (!model || !vectorizer) {
        throw std::runtime_error("Model and vectorizer must be trained first");
    }
    
    // Preprocess the text
    std::string cleanText = preprocessor.preprocess(text);
    
    // Check if text was properly preprocessed
    if (cleanText.empty()) {
        std::cerr << "Warning: Text was completely filtered out during preprocessing" << std::endl;
        // Use original text as fallback
        cleanText = text;
    }
    
    // Extract features
    std::vector<std::vector<double>> features = vectorizer->transform({cleanText});
    
    // Make prediction
    int sentiment = model->predict(features)[0];
    
    // Get confidence scores
    std::vector<double> decisionValues = model->decisionFunction(features);
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
    
    // Return the result as a map
    std::map<std::string, std::string> result;
    result["text"] = text;
    result["clean_text"] = cleanText;
    result["sentiment"] = sentimentLabel;
    result["raw_score"] = std::to_string(rawScore);
    result["scaled_score"] = std::to_string(scaledScore);
    result["confidence"] = std::to_string(confidence);
    result["probability"] = std::to_string(probability);
    
    // Add some context about the prediction
    std::stringstream explanation;
    
    if (confidence > 8.0) {
        explanation << "The model is very confident ";
    } else if (confidence > 5.0) {
        explanation << "The model is confident ";
    } else if (confidence > 2.0) {
        explanation << "The model suggests ";
    } else {
        explanation << "The model slightly leans toward ";
    }
    
    explanation << "that this text is " 
               << ((rawScore > 0) ? "positive" : "negative") 
               << " (score: " << std::fixed << std::setprecision(2) << scaledScore << ").";
    
    result["explanation"] = explanation.str();
    
    return result;
}

} // namespace nlp
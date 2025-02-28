#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <unordered_set>
#include <unordered_map>

namespace nlp {

/**
 * @brief Handles text preprocessing steps for sentiment analysis.
 * Includes methods for cleaning and normalizing text data.
 */
class TextPreprocessor {
public:
    TextPreprocessor();
    
    /**
     * Apply a series of preprocessing steps to clean and normalize text.
     * 
     * @param text The input text to preprocess
     * @param steps List of preprocessing steps to apply. If empty, all steps are applied.
     * @return The preprocessed text
     */
    std::string preprocess(const std::string& text, const std::vector<std::string>& steps = {});
    
private:
    // Set of stopwords to remove during preprocessing
    std::unordered_set<std::string> stops;
    
    // Preprocessing step implementations
    std::string removeNonAscii(const std::string& text);
    std::string lowercase(const std::string& text);
    std::string removePunctuation(const std::string& text);
    std::string removeNumbers(const std::string& text);
    std::string stripWhitespace(const std::string& text);
    std::string removeStopwords(const std::string& text);
    std::string stemWords(const std::string& text);
};

/**
 * @brief Feature representation for text using TF-IDF.
 */
class TfidfVectorizer {
public:
    TfidfVectorizer(bool sublinearTf = true, double maxDf = 0.5, 
                   size_t maxFeatures = 6228);
    
    /**
     * Fit the vectorizer on the corpus and transform it to TF-IDF features.
     * 
     * @param texts Vector of preprocessed text documents
     * @return Vector of vectors representing TF-IDF features
     */
    std::vector<std::vector<double>> fitTransform(const std::vector<std::string>& texts);
    
    /**
     * Transform new texts to TF-IDF features using the fitted vocabulary.
     * 
     * @param texts Vector of preprocessed text documents
     * @return Vector of vectors representing TF-IDF features
     */
    std::vector<std::vector<double>> transform(const std::vector<std::string>& texts);
    
private:
    bool sublinearTf;
    double maxDf;
    size_t maxFeatures;
    
    // Vocabulary and document frequencies
    std::unordered_map<std::string, size_t> vocabulary;
    std::unordered_map<std::string, size_t> documentFrequencies;
    size_t documentCount;
    
    // Helper methods
    std::vector<std::string> tokenize(const std::string& text);
    void buildVocabulary(const std::vector<std::string>& texts);
    std::unordered_map<std::string, double> computeTfIdf(const std::string& text);
};

/**
 * @brief SGD Classifier for sentiment analysis.
 */
class SGDClassifier {
public:
    SGDClassifier(const std::string& loss = "modified_huber", 
                 const std::string& learningRate = "adaptive",
                 const std::string& penalty = "elasticnet",
                 double alpha = 2.9e-05, 
                 double eta0 = 0.00164);
    
    /**
     * Train the classifier on the given features and labels.
     * 
     * @param X Training feature matrix
     * @param y Training labels
     */
    void fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y);
    
    /**
     * Predict sentiment labels for the given features.
     * 
     * @param X Feature matrix to predict
     * @return Vector of predicted labels
     */
    std::vector<int> predict(const std::vector<std::vector<double>>& X);
    
    /**
     * Calculate the decision function values for the given features.
     * 
     * @param X Feature matrix
     * @return Vector of decision function values
     */
    std::vector<double> decisionFunction(const std::vector<std::vector<double>>& X);
    
    /**
     * Calculate the accuracy score of the model on the given features and labels.
     * 
     * @param X Test feature matrix
     * @param y Test labels
     * @return The accuracy score
     */
    double score(const std::vector<std::vector<double>>& X, const std::vector<int>& y);
    
private:
    std::string loss;
    std::string learningRate;
    std::string penalty;
    double alpha;
    double eta0;
    
    std::vector<double> weights;
    double intercept;
    
    // Helper methods
    double predict_proba(const std::vector<double>& x);
};

/**
 * @brief Helper class for confusion matrix and classification reporting.
 */
class ModelEvaluator {
public:
    /**
     * Generate confusion matrix from predicted and actual labels.
     * 
     * @param yTrue Actual labels
     * @param yPred Predicted labels
     * @return Confusion matrix as a 2D vector
     */
    static std::vector<std::vector<int>> confusionMatrix(const std::vector<int>& yTrue, 
                                                       const std::vector<int>& yPred);
    
    /**
     * Generate classification report including precision, recall, and F1-score.
     * 
     * @param yTrue Actual labels
     * @param yPred Predicted labels
     * @param targetNames Names of the classes
     * @return Classification report as a string
     */
    static std::string classificationReport(const std::vector<int>& yTrue, 
                                          const std::vector<int>& yPred,
                                          const std::vector<std::string>& targetNames);
};

/**
 * @brief Class for generating ASCII-based word clouds
 */
class AsciiWordCloud {
public:
    /**
     * @brief Generate a word cloud from text with a specific sentiment
     * 
     * @param texts Collection of texts with the specified sentiment
     * @param maxWords Maximum number of words to include in the cloud
     * @param width Width of the ASCII display
     * @param height Height of the ASCII display
     * @param isPositive Whether this is a positive sentiment cloud (affects colors)
     * @return std::string Formatted ASCII word cloud
     */
    static std::string generateWordCloud(
        const std::vector<std::string>& texts, 
        size_t maxWords = 30, 
        size_t width = 80, 
        size_t height = 15,
        bool isPositive = true);
    
    /**
     * @brief Output a word cloud to stdout with colors
     * 
     * @param texts Collection of texts with the specified sentiment
     * @param maxWords Maximum number of words to include in the cloud
     * @param width Width of the ASCII display
     * @param height Height of the ASCII display
     * @param isPositive Whether this is a positive sentiment cloud (affects colors)
     */
    static void displayWordCloud(
        const std::vector<std::string>& texts, 
        size_t maxWords = 30, 
        size_t width = 80, 
        size_t height = 15,
        bool isPositive = true);

private:
    /**
     * @brief Count word frequencies in a collection of texts
     * 
     * @param texts Collection of texts
     * @return std::unordered_map<std::string, int> Word frequency map
     */
    static std::unordered_map<std::string, int> countWordFrequencies(
        const std::vector<std::string>& texts);
    
    /**
     * @brief Get a list of top words by frequency
     * 
     * @param wordFreqs Word frequency map
     * @param maxWords Maximum number of words to include
     * @return std::vector<std::pair<std::string, int>> List of (word, frequency) pairs
     */
    static std::vector<std::pair<std::string, int>> getTopWords(
        const std::unordered_map<std::string, int>& wordFreqs, 
        size_t maxWords);
    
    /**
     * @brief Format a word with ASCII art based on its frequency
     * 
     * @param word The word to format
     * @param freq The word's frequency
     * @param maxFreq The maximum frequency in the set
     * @param isPositive Whether this is for a positive sentiment cloud
     * @return std::string Formatted word with ASCII art
     */
    static std::string formatWord(
        const std::string& word, 
        int freq, 
        int maxFreq,
        bool isPositive);

    /**
     * @brief Get ANSI color code for word based on frequency and sentiment
     * 
     * @param freq Word frequency
     * @param maxFreq Maximum frequency in set
     * @param isPositive Whether this is for a positive sentiment cloud
     * @return std::string ANSI color code
     */
    static std::string getColorCode(int freq, int maxFreq, bool isPositive);
    
    /**
     * @brief Reset ANSI color formatting
     * 
     * @return std::string ANSI reset code
     */
    static std::string resetColor();
};

/**
 * @brief Main class for performing sentiment analysis.
 * Handles data loading, preprocessing, visualization, and model training.
 */
class SentimentAnalyzer {
public:
    SentimentAnalyzer();
    
    /**
     * Load tweet dataset from a CSV file.
     * 
     * @param dataPath Path to the CSV file
     * @return True if loading was successful
     */
    bool loadData(const std::string& dataPath);
    
    /**
     * Preprocess the tweets in the dataset.
     * 
     * @param steps List of preprocessing steps to apply
     * @return True if preprocessing was successful
     */
    bool preprocessData(const std::vector<std::string>& steps = {});
    
    /**
     * Generate a word cloud for tweets with the specified sentiment.
     * 
     * @param sentimentValue The sentiment label value (0 for negative, 4 for positive)
     * @param outputPath Path to save the word cloud text (optional)
     * @return True if word cloud generation was successful
     */
    bool generateWordCloud(int sentimentValue, const std::string& outputPath = "");
    
    /**
     * Extract TF-IDF features from preprocessed tweets.
     * 
     * @param maxDf Ignore terms that appear in more than this fraction of documents
     * @param maxFeatures Only use top max_features terms ordered by frequency
     * @return True if feature extraction was successful
     */
    bool extractFeatures(double maxDf = 0.5, size_t maxFeatures = 6228);
    
    /**
     * Split the dataset into training and testing sets.
     * 
     * @param testSize Size of the test set
     * @param randomState Random seed for reproducibility
     * @return True if splitting was successful
     */
    bool splitData(size_t testSize = 40000, unsigned int randomState = 42);
    
    /**
     * Train a sentiment classifier using SGD.
     * 
     * @param alpha Regularization parameter
     * @param eta0 Initial learning rate
     * @return True if training was successful
     */
    bool trainModel(double alpha = 2.9e-05, double eta0 = 0.00164);
    
    /**
     * Evaluate the trained sentiment classifier.
     * 
     * @return True if evaluation was successful
     */
    bool evaluateModel();
    
    /**
     * Predict the sentiment of new text.
     * 
     * @param text The text to analyze
     * @return Map containing the prediction and confidence
     */
    std::map<std::string, std::string> predictSentiment(const std::string& text);
    
private:
    TextPreprocessor preprocessor;
    std::unique_ptr<TfidfVectorizer> vectorizer;
    std::unique_ptr<SGDClassifier> model;
    
    // Dataset
    std::vector<std::pair<int, std::string>> data;
    std::vector<std::string> cleanedTexts;
    
    // Features and labels
    std::vector<std::vector<double>> features;
    std::vector<int> labels;
    
    // Train-test split
    std::vector<std::vector<double>> xTrain;
    std::vector<std::vector<double>> xTest;
    std::vector<int> yTrain;
    std::vector<int> yTest;
};

} // namespace nlp
/**
 * @file sentiment_analysis.h
 * @brief Main header file for the sentiment analysis library
 * 
 * This header defines the core functionality for sentiment analysis,
 * including text preprocessing, feature extraction, classification,
 * and visualization. The library is designed with a functional approach
 * and modern C++ features.
 */

#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <unordered_set>
#include <unordered_map>
#include <optional>
#include <variant>
#include <functional>
#include <concepts>
#include <ranges>
#include <span>
#include <algorithm>

namespace nlp {

/**
 * @brief Concept for string-like types that can be processed
 * 
 * This concept allows our functions to accept any type that can be
 * converted to a string_view, making the API more flexible.
 */
template <typename T>
concept StringLike = std::convertible_to<T, std::string_view>;

/**
 * @brief Concept for text processors
 * 
 * This concept defines the requirements for a text processor,
 * which must be able to process a string and return another string.
 */
template <typename T>
concept TextProcessor = requires(T t, const std::string& s) {
    { t.process(s) } -> std::convertible_to<std::string>;
};

/**
 * @brief Handles text preprocessing steps for sentiment analysis
 * 
 * TextPreprocessor provides methods to clean and normalize text data
 * before feature extraction and classification. It removes noise,
 * standardizes formatting, and prepares text for analysis.
 */
class TextPreprocessor {
public:
    /**
     * @brief Constructor that initializes stopwords
     */
    TextPreprocessor();
    
    /**
     * @brief Apply preprocessing steps to clean and normalize text
     * 
     * This is the main method for text preprocessing. It applies a series
     * of transformations to the input text to prepare it for analysis.
     * 
     * @param text The input text to preprocess
     * @param steps List of preprocessing steps to apply. If empty, all steps are applied.
     * @return The preprocessed text
     */
    std::string preprocess(std::string_view text, 
                          const std::vector<std::string>& steps = {}) const;
    
    /**
     * @brief Type definition for preprocessing function
     * 
     * This defines a function type that takes a string_view and returns a string.
     */
    using PreprocessingFunc = std::function<std::string(std::string_view)>;
    
    /**
     * @brief Get all available preprocessing functions
     * 
     * Returns a map of all preprocessing functions by name, allowing
     * for dynamic selection of preprocessing steps.
     * 
     * @return Map of preprocessing function names to function objects
     */
    std::unordered_map<std::string, PreprocessingFunc> getPreprocessingFunctions() const;
    
private:
    // Set of stopwords to remove during preprocessing
    std::unordered_set<std::string> stops;
    
    // Individual preprocessing steps as member functions
    std::string removeNonAscii(std::string_view text) const;
    std::string lowercase(std::string_view text) const;
    std::string removePunctuation(std::string_view text) const;
    std::string removeNumbers(std::string_view text) const;
    std::string stripWhitespace(std::string_view text) const;
    std::string removeStopwords(std::string_view text) const;
    std::string stemWords(std::string_view text) const;
};

/**
 * @brief Feature representation for text using TF-IDF
 * 
 * TfidfVectorizer converts a collection of text documents into numerical
 * feature vectors using the Term Frequency-Inverse Document Frequency 
 * (TF-IDF) approach. This transforms text into a format suitable for
 * machine learning algorithms.
 */
class TfidfVectorizer {
public:
    /**
     * @brief Constructor with configurable parameters
     * 
     * @param sublinearTf Whether to use sublinear term frequency scaling
     * @param maxDf Maximum document frequency threshold for terms
     * @param maxFeatures Maximum number of features to extract
     */
    TfidfVectorizer(bool sublinearTf = true, double maxDf = 0.5, 
                   size_t maxFeatures = 6228);
    
    /**
     * @brief Fit the vectorizer on a corpus and transform it to TF-IDF features
     * 
     * This method learns the vocabulary from the corpus and transforms
     * the documents into TF-IDF feature vectors in one step.
     * 
     * @param texts Vector of text documents
     * @return Matrix of TF-IDF features (one row per document)
     */
    template <StringLike T>
    std::vector<std::vector<double>> fitTransform(const std::vector<T>& texts) {
        // Convert texts to string_view for processing
        std::vector<std::string_view> text_views;
        text_views.reserve(texts.size());
        
        for (const auto& text : texts) {
            text_views.emplace_back(text);
        }
        
        // Build vocabulary
        buildVocabulary(text_views);
        
        // Then transform
        return transform(text_views);
    }
    
    /**
     * @brief Transform documents to TF-IDF features using the fitted vocabulary
     * 
     * This method transforms text documents into TF-IDF feature vectors
     * using the vocabulary learned during the fitTransform step.
     * 
     * @param texts Vector of text documents
     * @return Matrix of TF-IDF features (one row per document)
     */
    template <StringLike T>
    std::vector<std::vector<double>> transform(const std::vector<T>& texts) const {
        std::vector<std::vector<double>> result(texts.size(), std::vector<double>(vocabulary.size(), 0.0));
        
        for (size_t i = 0; i < texts.size(); ++i) {
            auto tfidfScores = computeTfIdf(std::string_view(texts[i]));
            
            for (const auto& [term, score] : tfidfScores) {
                auto it = vocabulary.find(term);
                if (it != vocabulary.end()) {
                    result[i][it->second] = score;
                }
            }
        }
        
        return result;
    }
    
    /**
     * @brief Get the size of the vocabulary
     * 
     * @return Number of terms in the vocabulary
     */
    size_t getVocabularySize() const { return vocabulary.size(); }
    
private:
    bool sublinearTf;
    double maxDf;
    size_t maxFeatures;
    
    // Vocabulary and document frequencies
    std::unordered_map<std::string, size_t> vocabulary;
    std::unordered_map<std::string, size_t> documentFrequencies;
    size_t documentCount;
    
    // Helper methods
    std::vector<std::string> tokenize(std::string_view text) const;
    void buildVocabulary(const std::vector<std::string_view>& texts);
    std::unordered_map<std::string, double> computeTfIdf(std::string_view text) const;
};

/**
 * @brief Base class for classification models
 * 
 * This abstract class defines the interface for all classification models.
 * Any concrete classifier must implement these methods.
 */
class ClassifierModel {
public:
    /**
     * @brief Virtual destructor for proper cleanup of derived classes
     */
    virtual ~ClassifierModel() = default;
    
    /**
     * @brief Train the classifier on the given features and labels
     * 
     * @param X Training feature matrix
     * @param y Training labels
     */
    virtual void fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y) = 0;
    
    /**
     * @brief Predict sentiment labels for the given features
     * 
     * @param X Feature matrix to predict
     * @return Vector of predicted labels
     */
    virtual std::vector<int> predict(const std::vector<std::vector<double>>& X) const = 0;
    
    /**
     * @brief Calculate the decision function values for the given features
     * 
     * @param X Feature matrix
     * @return Vector of decision function values
     */
    virtual std::vector<double> decisionFunction(const std::vector<std::vector<double>>& X) const = 0;
    
    /**
     * @brief Calculate the accuracy score of the model
     * 
     * @param X Test feature matrix
     * @param y Test labels
     * @return The accuracy score
     */
    virtual double score(const std::vector<std::vector<double>>& X, const std::vector<int>& y) const = 0;
};

/**
 * @brief SGD Classifier for sentiment analysis
 * 
 * This class implements a Stochastic Gradient Descent classifier
 * for sentiment analysis, supporting various loss functions and
 * regularization techniques.
 */
class SGDClassifier : public ClassifierModel {
public:
    /**
     * @brief Constructor with configurable parameters
     * 
     * @param loss Loss function to use
     * @param learningRate Learning rate schedule
     * @param penalty Regularization type
     * @param alpha Regularization strength
     * @param eta0 Initial learning rate
     */
    SGDClassifier(const std::string& loss = "modified_huber", 
                 const std::string& learningRate = "adaptive",
                 const std::string& penalty = "elasticnet",
                 double alpha = 2.9e-05, 
                 double eta0 = 0.00164);
    
    /**
     * @brief Train the classifier on the given features and labels
     * 
     * @param X Training feature matrix
     * @param y Training labels
     */
    void fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y) override;
    
    /**
     * @brief Predict sentiment labels for the given features
     * 
     * @param X Feature matrix to predict
     * @return Vector of predicted labels
     */
    std::vector<int> predict(const std::vector<std::vector<double>>& X) const override;
    
    /**
     * @brief Calculate the decision function values for the given features
     * 
     * @param X Feature matrix
     * @return Vector of decision function values
     */
    std::vector<double> decisionFunction(const std::vector<std::vector<double>>& X) const override;
    
    /**
     * @brief Calculate the accuracy score of the model
     * 
     * @param X Test feature matrix
     * @param y Test labels
     * @return The accuracy score
     */
    double score(const std::vector<std::vector<double>>& X, const std::vector<int>& y) const override;
    
    /**
     * @brief Calculate probability estimate for a feature vector
     * 
     * @param x Feature vector
     * @return Probability estimate between 0 and 1
     */
    double predict_proba(const std::vector<double>& x) const;
    
private:
    std::string loss;
    std::string learningRate;
    std::string penalty;
    double alpha;
    double eta0;
    
    std::vector<double> weights;
    double intercept;
    
    // Helper methods for implementing the SGD algorithm
    // These would be defined in the implementation file
};

/**
 * @brief Helper class for model evaluation
 * 
 * This class provides static methods for evaluating classification models,
 * including confusion matrices, classification reports, and performance metrics.
 */
class ModelEvaluator {
public:
    /**
     * @brief Generate confusion matrix from predicted and actual labels
     * 
     * @param yTrue Actual labels
     * @param yPred Predicted labels
     * @return Confusion matrix as a 2D vector
     */
    static std::vector<std::vector<int>> confusionMatrix(std::span<const int> yTrue, 
                                                       std::span<const int> yPred);
    
    /**
     * @brief Generate classification report with precision, recall, and F1-score
     * 
     * @param yTrue Actual labels
     * @param yPred Predicted labels
     * @param targetNames Names of the classes
     * @return Classification report as a string
     */
    static std::string classificationReport(std::span<const int> yTrue, 
                                          std::span<const int> yPred,
                                          const std::vector<std::string>& targetNames);
    
    /**
     * @brief Calculate performance metrics for a classification model
     * 
     * @param yTrue Actual labels
     * @param yPred Predicted labels
     * @return Map containing various metrics (accuracy, precision, recall, f1)
     */
    static std::unordered_map<std::string, double> calculateMetrics(std::span<const int> yTrue, 
                                                                  std::span<const int> yPred);
};

/**
 * @brief Class for generating ASCII-based word clouds
 * 
 * This class provides methods for visualizing word frequencies
 * as ASCII art word clouds, with support for color and custom formatting.
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
     * @return Formatted ASCII word cloud
     */
    static std::string generateWordCloud(
        std::span<const std::string> texts, 
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
        std::span<const std::string> texts, 
        size_t maxWords = 30, 
        size_t width = 80, 
        size_t height = 15,
        bool isPositive = true);

    /**
     * @brief Configuration options for word cloud visualization
     */
    struct CloudConfig {
        size_t maxWords = 30;       // Maximum number of words to include
        size_t width = 80;          // Width of ASCII display
        size_t height = 15;         // Height of ASCII display
        bool useColor = true;       // Whether to use ANSI colors
        bool useBars = true;        // Whether to display frequency bars
        bool showFrequencies = true; // Whether to show word frequencies
    };
    
    /**
     * @brief Generate a word cloud with custom configuration
     * 
     * @param texts Collection of texts
     * @param config Configuration options
     * @param isPositive Whether this is a positive sentiment (affects colors)
     * @return Generated word cloud
     */
    static std::string generateCustomCloud(
        std::span<const std::string> texts,
        const CloudConfig& config,
        bool isPositive = true);

private:
    /**
     * @brief Count word frequencies in a collection of texts
     * 
     * Uses a recursive approach to process the texts.
     * 
     * @param texts Collection of texts
     * @return Word frequency map
     */
    static std::unordered_map<std::string, int> countWordFrequencies(
        std::span<const std::string> texts);
    
    /**
     * @brief Get a list of top words by frequency
     * 
     * @param wordFreqs Word frequency map
     * @param maxWords Maximum number of words to include
     * @return List of (word, frequency) pairs
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
     * @return Formatted word with ASCII art
     */
    static std::string formatWord(
        std::string_view word, 
        int freq, 
        int maxFreq,
        bool isPositive);

    /**
     * @brief Get ANSI color code for word based on frequency and sentiment
     * 
     * @param freq Word frequency
     * @param maxFreq Maximum frequency in set
     * @param isPositive Whether this is for a positive sentiment cloud
     * @return ANSI color code
     */
    static std::string getColorCode(int freq, int maxFreq, bool isPositive);
    
    /**
     * @brief Reset ANSI color formatting
     * 
     * @return ANSI reset code
     */
    static std::string resetColor();
};

/**
 * @brief Result of a sentiment prediction
 * 
 * This struct contains all the information about a sentiment prediction,
 * including the input text, cleaned text, predicted sentiment, and confidence.
 */
struct SentimentResult {
    std::string text;               // Original text
    std::string cleanText;          // Preprocessed text
    std::string sentiment;          // "Positive" or "Negative"
    double rawScore;                // Raw decision function score
    double scaledScore;             // Scaled decision score
    double confidence;              // Confidence value
    double probability;             // Probability estimate
    std::string explanation;        // Human-readable explanation
    
    /**
     * @brief Convert to a string map for serialization
     * 
     * @return Map representation of the result
     */
    std::map<std::string, std::string> toMap() const;
    
    /**
     * @brief Create a SentimentResult from a string map
     * 
     * @param map Map representation of a result
     * @return Reconstructed SentimentResult
     */
    static SentimentResult fromMap(const std::map<std::string, std::string>& map);
};

/**
 * @brief Dataset for sentiment analysis
 * 
 * This struct contains all the data needed for sentiment analysis,
 * including raw data, preprocessed texts, labels, and train/test splits.
 */
struct SentimentDataset {
    std::vector<std::pair<int, std::string>> data;     // (label, text) pairs
    std::vector<std::string> cleanedTexts;             // Preprocessed texts
    std::vector<int> labels;                           // Labels (0=negative, 4=positive)
    
    // Split indices
    std::vector<size_t> trainIndices;
    std::vector<size_t> testIndices;
    
    /**
     * @brief Get training texts based on trainIndices
     * 
     * @return Vector of training texts
     */
    std::vector<std::string> getTrainTexts() const;
    
    /**
     * @brief Get test texts based on testIndices
     * 
     * @return Vector of test texts
     */
    std::vector<std::string> getTestTexts() const;
    
    /**
     * @brief Get training labels based on trainIndices
     * 
     * @return Vector of training labels
     */
    std::vector<int> getTrainLabels() const;
    
    /**
     * @brief Get test labels based on testIndices
     * 
     * @return Vector of test labels
     */
    std::vector<int> getTestLabels() const;
    
    /**
     * @brief Get texts with a specific sentiment
     * 
     * @param sentiment Sentiment value (0=negative, 4=positive)
     * @return Vector of texts with the specified sentiment
     */
    std::vector<std::string> getTextsWithSentiment(int sentiment) const;
};

/**
 * @brief Main class for performing sentiment analysis
 * 
 * This class orchestrates the entire sentiment analysis process,
 * from data loading and preprocessing to model training and evaluation.
 */
class SentimentAnalyzer {
public:
    /**
     * @brief Constructor
     */
    SentimentAnalyzer();
    
    /**
     * @brief Load tweet dataset from a CSV file
     * 
     * @param dataPath Path to the CSV file
     * @return Result status wrapped in optional (empty if failed)
     */
    std::optional<SentimentDataset> loadData(const std::string& dataPath);
    
    /**
     * @brief Preprocess the tweets in the dataset
     * 
     * @param dataset The dataset to preprocess
     * @param steps List of preprocessing steps to apply
     * @return Preprocessed dataset
     */
    SentimentDataset preprocessData(SentimentDataset dataset, 
                                  const std::vector<std::string>& steps = {});
    
    /**
     * @brief Generate a word cloud for tweets with the specified sentiment
     * 
     * @param dataset The sentiment dataset
     * @param sentimentValue The sentiment label value (0 for negative, 4 for positive)
     * @param outputPath Path to save the word cloud text (optional)
     * @return True if word cloud generation was successful
     */
    bool generateWordCloud(const SentimentDataset& dataset, 
                         int sentimentValue, 
                         const std::string& outputPath = "");
    
    /**
     * @brief Extract TF-IDF features from preprocessed tweets
     * 
     * @param dataset The preprocessed dataset
     * @param maxDf Ignore terms that appear in more than this fraction of documents
     * @param maxFeatures Only use top max_features terms ordered by frequency
     * @return Pair of (features, labels)
     */
    std::pair<std::vector<std::vector<double>>, std::vector<int>> 
    extractFeatures(const SentimentDataset& dataset, 
                  double maxDf = 0.5, 
                  size_t maxFeatures = 6228);
    
    /**
     * @brief Split the dataset into training and testing sets
     * 
     * @param dataset The dataset to split
     * @param testSize Size of the test set
     * @param randomState Random seed for reproducibility
     * @return Dataset with train/test split indices
     */
    SentimentDataset splitData(SentimentDataset dataset, 
                             size_t testSize = 40000, 
                             unsigned int randomState = 42);
    
    /**
     * @brief Train a sentiment classifier
     * 
     * @param X_train Training feature matrix
     * @param y_train Training labels
     * @param alpha Regularization parameter
     * @param eta0 Initial learning rate
     * @return Trained classifier model
     */
    std::unique_ptr<ClassifierModel> trainModel(
        const std::vector<std::vector<double>>& X_train,
        const std::vector<int>& y_train,
        double alpha = 2.9e-05, 
        double eta0 = 0.00164);
    
    /**
     * @brief Evaluate the trained sentiment classifier
     * 
     * @param model The trained model to evaluate
     * @param X_test Test feature matrix
     * @param y_test Test labels
     * @return Evaluation metrics
     */
    std::unordered_map<std::string, double> evaluateModel(
        const ClassifierModel& model,
        const std::vector<std::vector<double>>& X_test, 
        const std::vector<int>& y_test);
    
    /**
     * @brief Predict the sentiment of new text
     * 
     * @param text The text to analyze
     * @param model The trained model
     * @param vectorizer The trained vectorizer
     * @return Sentiment prediction result
     */
    SentimentResult predictSentiment(
        const std::string& text,
        const ClassifierModel& model,
        const TfidfVectorizer& vectorizer);
    
private:
    TextPreprocessor preprocessor;
    
    /**
     * @brief Generate a human-readable explanation of the sentiment prediction
     * 
     * @param score Decision function score
     * @param confidence Confidence value
     * @return Explanation text
     */
    std::string generateExplanation(double score, double confidence) const;
};

} // namespace nlp

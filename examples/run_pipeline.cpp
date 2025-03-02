/**
 * @file run_pipeline.cpp
 * @brief Complete pipeline for sentiment analysis
 * 
 * This example demonstrates the full sentiment analysis workflow,
 * from loading data to preprocessing, feature extraction, model training,
 * evaluation, and prediction on new examples.
 */

#include <iostream>
#include "sentiment_analysis.h"
#include "ascii_word_cloud.h"
#include "tfidf_vectorizer.h"
#include "sgd_classifier.h"
#include "model_evaluator.h"

/**
 * Main entry point for the sentiment analysis pipeline.
 * 
 * @param argc Command line argument count
 * @param argv Command line arguments
 * @return Exit status code
 */
int main(int argc, char* argv[]) {
    std::cout << "Running NLP Sentiment Analysis Pipeline...\n";

    // Initialize components
    nlp::SentimentAnalyzer analyzer;
    nlp::TfidfVectorizer vectorizer;
    nlp::SGDClassifier classifier;

    // Determine data path
    std::string dataPath = "../data/twitter_data.csv";
    if (argc > 1) {
        dataPath = argv[1];
    }

    // Sample size limit for large datasets
    const size_t SAMPLE_LIMIT = 1000;

    // Load dataset
    std::cout << "Loading dataset from " << dataPath << "...\n";
    auto dataset = analyzer.loadData(dataPath);
    if (!dataset) {
        std::cerr << "Error loading dataset!\n";
        return 1;
    }

    // Limit the dataset size if it's very large
    if (dataset->data.size() > SAMPLE_LIMIT) {
        std::cout << "Dataset is large. Using " << SAMPLE_LIMIT << " samples for analysis.\n";
        
        // Create a smaller dataset to work with
        std::vector<std::pair<int, std::string>> sampledData;
        size_t stride = dataset->data.size() / SAMPLE_LIMIT;
        
        for (size_t i = 0; i < dataset->data.size() && sampledData.size() < SAMPLE_LIMIT; i += stride) {
            sampledData.push_back(dataset->data[i]);
        }
        
        // Create a new dataset with the sampled data
        nlp::SentimentDataset sampledDataset(sampledData);
        *dataset = std::move(sampledDataset);
    }

    // Preprocess dataset
    std::cout << "Preprocessing dataset...\n";
    nlp::SentimentDataset processedData = analyzer.preprocessData(std::move(*dataset));

    // Generate word clouds
    std::cout << "Generating word clouds...\n";
    analyzer.generateWordCloud(processedData, 4, "positive_wordcloud.txt");
    analyzer.generateWordCloud(processedData, 0, "negative_wordcloud.txt");

    // Extract features
    std::cout << "Extracting features...\n";
    auto featurePair = analyzer.extractFeatures(processedData);

    // Store features in dataset
    processedData.features = featurePair.first;
    processedData.labels = featurePair.second;

    // Split data into training & test sets
    std::cout << "Splitting dataset...\n";
    auto splitDataset = analyzer.splitData(std::move(processedData));

    // Get train/test data
    auto X_train = splitDataset.getTrainFeatures();
    auto X_test = splitDataset.getTestFeatures();
    auto y_train = splitDataset.getTrainLabels();
    auto y_test = splitDataset.getTestLabels();

    // Train model
    std::cout << "Training model...\n";
    auto model = analyzer.trainModel(X_train, y_train);

    // Evaluate model
    std::cout << "Evaluating model...\n";
    auto metrics = analyzer.evaluateModel(*model, X_test, y_test);
    std::cout << "Model Accuracy: " << metrics["accuracy"] << std::endl;

    // Example predictions
    std::cout << "Running example predictions...\n";
    std::vector<std::string> examples = {
        "I absolutely love this new product! It's amazing!",
        "This is the worst experience I've ever had. Terrible service.",
        "The weather is quite nice today, isn't it?",
        "I'm not sure how I feel about this movie."
    };

    // Fit vectorizer on training texts for consistent features
    vectorizer.fitTransform(splitDataset.getTrainTexts());

    // Make predictions on examples
    for (const auto& text : examples) {
        auto result = analyzer.predictSentiment(text, *model, vectorizer);
        std::cout << "Text: " << text << "\nSentiment: " << result.sentiment 
                  << " (confidence: " << result.confidence << ")\n\n";
    }

    std::cout << "Pipeline complete!\n";
    return 0;
}

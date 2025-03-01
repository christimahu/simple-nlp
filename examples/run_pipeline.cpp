#include <iostream>
#include "sentiment_analysis.h"
#include "ascii_word_cloud.h"
#include "tfidf_vectorizer.h"
#include "sgd_classifier.h"
#include "model_evaluator.h"

int main() {
    std::cout << "Running NLP Sentiment Analysis Pipeline...\n";

    // Initialize components
    nlp::SentimentAnalyzer analyzer;
    nlp::TfidfVectorizer vectorizer;
    nlp::SGDClassifier classifier;

    // Load dataset
    std::cout << "Loading dataset...\n";
    auto dataset = analyzer.loadData("../data/twitter_data.csv");
    if (!dataset) {
        std::cerr << "Error loading dataset!\n";
        return 1;
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
    auto features = analyzer.extractFeatures(processedData);

    // Split data into training & test sets
    std::cout << "Splitting dataset...\n";
    auto [X_train, X_test, y_train, y_test] = analyzer.splitData(std::move(features));

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

    for (const auto& text : examples) {
        auto result = analyzer.predictSentiment(text, *model, vectorizer);
        std::cout << "Text: " << text << "\nSentiment: " << result.label << "\n\n";
    }

    std::cout << "Pipeline complete!\n";
    return 0;
}


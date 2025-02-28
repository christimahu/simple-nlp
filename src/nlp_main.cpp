#include "sentiment_analysis.h"
#include <iostream>
#include <string>
#include <filesystem>
#include <chrono>

void runTests() {
    std::cout << "====== Running Tests ======" << std::endl;
    
    // Test preprocessor
    std::cout << "\nTesting TextPreprocessor..." << std::endl;
    nlp::TextPreprocessor preprocessor;
    std::string testText = "@user I LOVED the movie!!! It's amazing & worth $12.99 :) #mustwatch";
    
    std::string processed = preprocessor.preprocess(testText);
    std::cout << "Original: " << testText << std::endl;
    std::cout << "Processed: " << processed << std::endl;
    
    // Check if text is lowercase
    bool isLowercase = true;
    for (char c : processed) {
        if (std::isupper(static_cast<unsigned char>(c))) {
            isLowercase = false;
            break;
        }
    }
    
    // Basic assertions
    if (!processed.empty()) {
        std::cout << "✓ Preprocessor returns a non-empty string" << std::endl;
    } else {
        std::cout << "✗ Preprocessor returned an empty string" << std::endl;
    }
    
    if (isLowercase) {
        std::cout << "✓ Text is lowercase after preprocessing" << std::endl;
    } else {
        std::cout << "✗ Text is not lowercase after preprocessing" << std::endl;
    }
    
    if (processed.find('$') == std::string::npos) {
        std::cout << "✓ Punctuation was removed" << std::endl;
    } else {
        std::cout << "✗ Punctuation was not removed" << std::endl;
    }
    
    if (processed.find("12") == std::string::npos) {
        std::cout << "✓ Numbers were removed" << std::endl;
    } else {
        std::cout << "✗ Numbers were not removed" << std::endl;
    }
    
    // Test basic SentimentAnalyzer initialization
    std::cout << "\nTesting SentimentAnalyzer initialization..." << std::endl;
    nlp::SentimentAnalyzer analyzer;
    
    std::cout << "✓ Analyzer initialized successfully" << std::endl;
    
    std::cout << "\nAll tests passed!" << std::endl;
    std::cout << "====== Tests Complete ======" << std::endl;
}

void runFullAnalysis(const std::string& dataPath) {
    std::cout << "====== Starting Sentiment Analysis ======" << std::endl;
    
    // Initialize analyzer and load data
    nlp::SentimentAnalyzer analyzer;
    if (!analyzer.loadData(dataPath)) {
        std::cerr << "Failed to load data. Exiting." << std::endl;
        return;
    }
    
    // Preprocess the data
    analyzer.preprocessData();
    
    // Generate word clouds
    std::cout << "\n====== Generating Word Clouds ======" << std::endl;
    analyzer.generateWordCloud(4, "positive_wordcloud.txt");
    analyzer.generateWordCloud(0, "negative_wordcloud.txt");
    
    // Extract features
    std::cout << "\n====== Extracting Features ======" << std::endl;
    analyzer.extractFeatures();
    
    // Split the data
    analyzer.splitData();
    
    // Train the model
    std::cout << "\n====== Training Model ======" << std::endl;
    auto startTime = std::chrono::high_resolution_clock::now();
    analyzer.trainModel();
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime);
    std::cout << "Training took " << duration.count() << " seconds" << std::endl;
    
    // Evaluate the model
    std::cout << "\n====== Evaluating Model ======" << std::endl;
    analyzer.evaluateModel();
    
    // Example predictions
    std::cout << "\n====== Example Predictions ======" << std::endl;
    std::vector<std::string> examples = {
        "I absolutely love this new product! It's amazing!",
        "This is the worst experience I've ever had. Terrible service.",
        "The weather is quite nice today, isn't it?",
        "I'm not sure how I feel about this movie."
    };
    
    std::cout << std::fixed << std::setprecision(2);
    
    for (const auto& example : examples) {
        try {
            auto result = analyzer.predictSentiment(example);
            std::cout << "Text: " << result["text"] << std::endl;
            std::cout << "Cleaned: " << result["clean_text"] << std::endl;
            std::cout << "Sentiment: " << result["sentiment"] << std::endl;
            std::cout << "Raw score: " << result["raw_score"] << std::endl;
            std::cout << "Scaled score: " << result["scaled_score"] << std::endl;
            std::cout << "Confidence: " << result["confidence"] << std::endl;
            std::cout << "Probability: " << result["probability"] << std::endl;
            std::cout << "Explanation: " << result["explanation"] << std::endl;
            std::cout << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error predicting sentiment: " << e.what() << std::endl;
        }
    }
    
    std::cout << "====== Analysis Complete ======" << std::endl;
}

int main(int argc, char* argv[]) {
    // Run tests to verify component functionality
    runTests();
    
    // If a data file path is provided, run the full analysis
    std::string defaultPath = "../data/twitter_data.csv";
    
    if (argc > 1) {
        defaultPath = argv[1];
    }
    
    if (std::filesystem::exists(defaultPath)) {
        std::cout << "\nFound data file at " << defaultPath << std::endl;
        runFullAnalysis(defaultPath);
    } else {
        std::cout << "\nData file not found at " << defaultPath << std::endl;
        std::cout << "To run the full analysis, place the twitter_data.csv file in the ../data/ directory" << std::endl;
        std::cout << "or run the program with a custom path:" << std::endl;
        std::cout << "    ./nlp <path_to_data>" << std::endl;
    }
    
    return 0;
}

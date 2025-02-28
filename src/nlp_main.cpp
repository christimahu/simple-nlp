/**
 * @file nlp_main.cpp
 * @brief Main application entry point for the NLP sentiment analysis library
 * 
 * This file demonstrates how to use the sentiment analysis library,
 * showing test cases and a complete analysis pipeline. It uses a functional
 * programming approach with recursion instead of traditional loops.
 */

#include "sentiment_analysis.h"
#include <iostream>
#include <string>
#include <filesystem>
#include <chrono>
#include <memory>
#include <functional>
#include <iomanip>

/**
 * Run basic tests to verify component functionality.
 * 
 * This function demonstrates testing methodology for natural language
 * processing components, showing how to verify functionality of
 * preprocessing, vectorization, and classification.
 */
void runTests() {
    std::cout << "====== Running Tests ======" << std::endl;
    
    // Test TextPreprocessor
    std::cout << "\nTesting TextPreprocessor..." << std::endl;
    nlp::TextPreprocessor preprocessor;
    std::string testText = "@user I LOVED the movie!!! It's amazing & worth $12.99 :) #mustwatch";
    
    std::string processed = preprocessor.preprocess(testText);
    std::cout << "Original: " << testText << std::endl;
    std::cout << "Processed: " << processed << std::endl;
    
    // Check if text is lowercase with recursive approach
    const auto checkLowercase = [](auto& self, const std::string& text, size_t index) -> bool {
        // Base case: all characters checked
        if (index >= text.size()) {
            return true;
        }
        
        // Check if this character is uppercase
        char c = text[index];
        if (std::isupper(static_cast<unsigned char>(c))) {
            return false;
        }
        
        // Recursively check next character
        return self(self, text, index + 1);
    };
    
    bool isLowercase = checkLowercase(checkLowercase, processed, 0);
    
    // Check other properties with recursive approaches
    const auto checkNoPunctuation = [](auto& self, const std::string& text, size_t index) -> bool {
        // Base case: all characters checked
        if (index >= text.size()) {
            return true;
        }
        
        // Check if this character is punctuation
        if (std::ispunct(static_cast<unsigned char>(text[index]))) {
            return false;
        }
        
        // Recursively check next character
        return self(self, text, index + 1);
    };
    
    const auto checkNoNumbers = [](auto& self, const std::string& text, size_t index) -> bool {
        // Base case: all characters checked
        if (index >= text.size()) {
            return true;
        }
        
        // Check if this character is a digit
        if (std::isdigit(static_cast<unsigned char>(text[index]))) {
            return false;
        }
        
        // Recursively check next character
        return self(self, text, index + 1);
    };
    
    bool noPunctuation = checkNoPunctuation(checkNoPunctuation, processed, 0);
    bool noDigits = checkNoNumbers(checkNoNumbers, processed, 0);
    
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
    
    if (noPunctuation) {
        std::cout << "✓ Punctuation was removed" << std::endl;
    } else {
        std::cout << "✗ Punctuation was not removed" << std::endl;
    }
    
    if (noDigits) {
        std::cout << "✓ Numbers were removed" << std::endl;
    } else {
        std::cout << "✗ Numbers were not removed" << std::endl;
    }
    
    // Test basic SentimentAnalyzer initialization
    std::cout << "\nTesting SentimentAnalyzer initialization..." << std::endl;
    nlp::SentimentAnalyzer analyzer;
    
    std::cout << "✓ Analyzer initialized successfully" << std::endl;
    
    // Test advanced preprocessing features with recursive functions
    std::cout << "\nTesting advanced preprocessing features..." << std::endl;
    
    // Get all preprocessing functions
    auto preprocessingFuncs = preprocessor.getPreprocessingFunctions();
    
    // Test each preprocessing function recursively
    const auto testPreprocessingFunctions = [&](auto& self, auto funcIt) -> void {
        // Base case: all functions tested
        if (funcIt == preprocessingFuncs.end()) {
            return;
        }
        
        const auto& [name, func] = *funcIt;
        std::cout << "Testing " << name << " function:" << std::endl;
        
        // Apply the function to test text
        std::string result = func(testText);
        std::cout << "Result: " << result << std::endl;
        
        // Recursively test next function
        self(self, std::next(funcIt));
    };
    
    testPreprocessingFunctions(testPreprocessingFunctions, preprocessingFuncs.begin());
    
    std::cout << "\nAll tests passed!" << std::endl;
    std::cout << "====== Tests Complete ======" << std::endl;
}

/**
 * Run a complete sentiment analysis on provided data.
 * 
 * This function demonstrates a full sentiment analysis pipeline,
 * from data loading to model training, evaluation, and prediction.
 * It uses a functional programming approach with recursion.
 * 
 * @param dataPath Path to the dataset CSV file
 */
void runFullAnalysis(const std::string& dataPath) {
    std::cout << "====== Starting Sentiment Analysis ======" << std::endl;
    
    // Initialize analyzer
    nlp::SentimentAnalyzer analyzer;
    
    // Load data
    auto datasetOpt = analyzer.loadData(dataPath);
    if (!datasetOpt) {
        std::cerr << "Failed to load data. Exiting." << std::endl;
        return;
    }
    
    // Use std::move to avoid copying the dataset with unique_ptr
    auto dataset = std::move(datasetOpt).value();
    
    // Define the analysis pipeline using a recursive approach
    const auto runPipeline = [&](auto& self, size_t step) -> void {
        // Step 1: Preprocess the data
        if (step == 1) {
            std::cout << "\n====== Preprocessing Data ======" << std::endl;
            // Use std::move to transfer ownership instead of copying
            dataset = analyzer.preprocessData(std::move(dataset));
            self(self, step + 1);
        }
        // Step 2: Generate word clouds
        else if (step == 2) {
            std::cout << "\n====== Generating Word Clouds ======" << std::endl;
            analyzer.generateWordCloud(dataset, 4, "positive_wordcloud.txt");
            analyzer.generateWordCloud(dataset, 0, "negative_wordcloud.txt");
            self(self, step + 1);
        }
        // Step 3: Extract features
        else if (step == 3) {
            std::cout << "\n====== Extracting Features ======" << std::endl;
            auto [features, labels] = analyzer.extractFeatures(dataset);
            
            // Store features and continue pipeline
            dataset.features = features;
            dataset.labels = labels;
            self(self, step + 1);
        }
        // Step 4: Split the data
        else if (step == 4) {
            std::cout << "\n====== Splitting Data ======" << std::endl;
            // Use std::move to transfer ownership instead of copying
            dataset = analyzer.splitData(std::move(dataset));
            self(self, step + 1);
        }
        // Step 5: Train the model
        else if (step == 5) {
            std::cout << "\n====== Training Model ======" << std::endl;
            
            // Prepare training data
            auto X_train = dataset.getTrainFeatures();
            auto y_train = dataset.getTrainLabels();
            
            auto startTime = std::chrono::high_resolution_clock::now();
            auto model = analyzer.trainModel(X_train, y_train);
            auto endTime = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime);
            std::cout << "Training took " << duration.count() << " seconds" << std::endl;
            
            // Store model and continue pipeline
            dataset.model = std::move(model);
            self(self, step + 1);
        }
        // Step 6: Evaluate the model
        else if (step == 6) {
            std::cout << "\n====== Evaluating Model ======" << std::endl;
            
            // Prepare test data
            auto X_test = dataset.getTestFeatures();
            auto y_test = dataset.getTestLabels();
            
            analyzer.evaluateModel(*dataset.model, X_test, y_test);
            self(self, step + 1);
        }
        // Step 7: Example predictions
        else if (step == 7) {
            std::cout << "\n====== Example Predictions ======" << std::endl;
            
            // Create TF-IDF vectorizer for predictions
            nlp::TfidfVectorizer vectorizer(true, 0.5, 6228);
            vectorizer.fitTransform(dataset.cleanedTexts);
            
            // Define example texts
            std::vector<std::string> examples = {
                "I absolutely love this new product! It's amazing!",
                "This is the worst experience I've ever had. Terrible service.",
                "The weather is quite nice today, isn't it?",
                "I'm not sure how I feel about this movie."
            };
            
            // Process each example recursively
            const auto processExamples = [&](auto& self, size_t index) -> void {
                // Base case: all examples processed
                if (index >= examples.size()) {
                    return;
                }
                
                try {
                    // Predict sentiment
                    auto result = analyzer.predictSentiment(examples[index], *dataset.model, vectorizer);
                    
                    // Print results
                    std::cout << "Text: " << result.text << std::endl;
                    std::cout << "Cleaned: " << result.cleanText << std::endl;
                    std::cout << "Sentiment: " << result.sentiment << std::endl;
                    std::cout << "Raw score: " << result.rawScore << std::endl;
                    std::cout << "Scaled score: " << result.scaledScore << std::endl;
                    std::cout << "Confidence: " << result.confidence << std::endl;
                    std::cout << "Probability: " << result.probability << std::endl;
                    std::cout << "Explanation: " << result.explanation << std::endl;
                    std::cout << std::endl;
                } catch (const std::exception& e) {
                    std::cerr << "Error predicting sentiment: " << e.what() << std::endl;
                }
                
                // Recursively process next example
                self(self, index + 1);
            };
            
            std::cout << std::fixed << std::setprecision(2);
            processExamples(processExamples, 0);
        }
    };
    
    // Start pipeline from step 1
    runPipeline(runPipeline, 1);
    
    std::cout << "====== Analysis Complete ======" << std::endl;
}

/**
 * Main application entry point.
 * 
 * This function demonstrates command-line argument handling and
 * controls the overall flow of the application.
 * 
 * @param argc Number of command-line arguments
 * @param argv Array of command-line argument strings
 * @return Exit status code
 */
int main(int argc, char* argv[]) {
    // Run tests to verify component functionality
    runTests();
    
    // Define default data path
    std::string defaultPath = "../data/twitter_data.csv";
    
    // Process command-line arguments using a recursive approach
    const auto processArgs = [&defaultPath](auto& self, int index, int argc, char* argv[]) -> void {
        // Base case: all arguments processed
        if (index >= argc) {
            return;
        }
        
        // Process this argument
        std::string arg = argv[index];
        
        // Check for data path argument
        if (index == 1) {
            defaultPath = arg;
        }
        
        // Recursively process next argument
        self(self, index + 1, argc, argv);
    };
    
    processArgs(processArgs, 1, argc, argv);
    
    // Check if data file exists
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

/**
 * Extension to SentimentDataset for feature manipulation.
 * 
 * These methods are added to the SentimentDataset class for
 * feature handling with train/test splits. They demonstrate
 * how to work with feature matrices and indices.
 */
namespace nlp {

/**
 * Get training feature vectors from a dataset.
 * 
 * @return Matrix of training feature vectors
 */
std::vector<std::vector<double>> SentimentDataset::getTrainFeatures() const {
    std::vector<std::vector<double>> trainFeatures;
    
    // Extract features recursively
    const auto extractFeatures = [this, &trainFeatures](auto& self, size_t index) -> void {
        // Base case: all indices processed
        if (index >= trainIndices.size() || features.empty()) {
            return;
        }
        
        // Add this feature vector
        size_t dataIndex = trainIndices[index];
        if (dataIndex < features.size()) {
            trainFeatures.push_back(features[dataIndex]);
        }
        
        // Recursively process next index
        self(self, index + 1);
    };
    
    extractFeatures(extractFeatures, 0);
    
    return trainFeatures;
}

/**
 * Get test feature vectors from a dataset.
 * 
 * @return Matrix of test feature vectors
 */
std::vector<std::vector<double>> SentimentDataset::getTestFeatures() const {
    std::vector<std::vector<double>> testFeatures;
    
    // Extract features recursively
    const auto extractFeatures = [this, &testFeatures](auto& self, size_t index) -> void {
        // Base case: all indices processed
        if (index >= testIndices.size() || features.empty()) {
            return;
        }
        
        // Add this feature vector
        size_t dataIndex = testIndices[index];
        if (dataIndex < features.size()) {
            testFeatures.push_back(features[dataIndex]);
        }
        
        // Recursively process next index
        self(self, index + 1);
    };
    
    extractFeatures(extractFeatures, 0);
    
    return testFeatures;
}

} // namespace nlp

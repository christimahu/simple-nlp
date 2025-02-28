/**
 * @file file_analyzer.cpp
 * @brief Example application for analyzing sentiment from text files
 * 
 * This example demonstrates how to use the sentiment analysis library
 * to process text files, extract sentiments, and generate reports.
 * It shows file I/O operations and sentiment analysis integration.
 */

#include "sentiment_analysis.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <chrono>
#include <iomanip>

/**
 * Process a single file and analyze its sentiment.
 * 
 * This function demonstrates how to read a text file,
 * analyze its sentiment, and generate a detailed report.
 * 
 * @param analyzer The sentiment analyzer to use
 * @param model The trained sentiment model
 * @param vectorizer The TF-IDF vectorizer
 * @param filePath Path to the text file
 * @return True if analysis was successful
 */
bool analyzeSingleFile(
    nlp::SentimentAnalyzer& analyzer, // Changed from const to non-const
    const nlp::ClassifierModel& model,
    const nlp::TfidfVectorizer& vectorizer,
    const std::string& filePath) {
    
    std::cout << "Analyzing file: " << filePath << std::endl;
    
    // Open and read the file
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filePath << std::endl;
        return false;
    }
    
    // Read file contents using recursion
    const auto readFileContents = [&file](auto& self, std::string content = "") -> std::string {
        std::string line;
        
        // Base case: no more lines to read
        if (!std::getline(file, line)) {
            return content;
        }
        
        // Add this line and recursively read next line
        if (!content.empty()) {
            content += "\n";
        }
        content += line;
        
        return self(self, content);
    };
    
    std::string fileContent = readFileContents(readFileContents);
    
    // Split content into paragraphs using recursion
    std::vector<std::string> paragraphs;
    
    const auto splitParagraphs = [](auto& self, const std::string& text, 
                               size_t start, std::vector<std::string>& results) -> void {
        // Base case: end of text
        if (start >= text.size()) {
            return;
        }
        
        // Find next double newline (paragraph boundary)
        size_t end = text.find("\n\n", start);
        if (end == std::string::npos) {
            // Last paragraph
            std::string paragraph = text.substr(start);
            if (!paragraph.empty()) {
                results.push_back(paragraph);
            }
            return;
        }
        
        // Extract this paragraph
        std::string paragraph = text.substr(start, end - start);
        if (!paragraph.empty()) {
            results.push_back(paragraph);
        }
        
        // Recursively process next paragraph
        self(self, text, end + 2, results);
    };
    
    splitParagraphs(splitParagraphs, fileContent, 0, paragraphs);
    
    // Analyze each paragraph recursively
    std::vector<nlp::SentimentResult> results;
    
    const auto analyzeParagraphs = [&](auto& self, size_t index) -> void {
        // Base case: all paragraphs analyzed
        if (index >= paragraphs.size()) {
            return;
        }
        
        // Analyze this paragraph
        if (!paragraphs[index].empty()) {
            results.push_back(analyzer.predictSentiment(paragraphs[index], model, vectorizer));
        }
        
        // Recursively analyze next paragraph
        self(self, index + 1);
    };
    
    analyzeParagraphs(analyzeParagraphs, 0);
    
    // Generate report
    std::cout << "\nSentiment Analysis Report for: " << std::filesystem::path(filePath).filename() << std::endl;
    std::cout << "===================================================" << std::endl;
    std::cout << "Total paragraphs analyzed: " << results.size() << std::endl;
    
    // Count sentiment distribution recursively
    int positiveCount = 0;
    int negativeCount = 0;
    
    const auto countSentiments = [&](auto& self, size_t index) -> void {
        // Base case: all results counted
        if (index >= results.size()) {
            return;
        }
        
        // Count this result
        if (results[index].sentiment == "Positive") {
            positiveCount++;
        } else {
            negativeCount++;
        }
        
        // Recursively count next result
        self(self, index + 1);
    };
    
    countSentiments(countSentiments, 0);
    
    // Calculate sentiment ratio
    double positiveRatio = results.empty() ? 0.0 : 
                          static_cast<double>(positiveCount) / results.size();
    
    std::cout << "Positive paragraphs: " << positiveCount << " (" 
              << std::fixed << std::setprecision(1) << (positiveRatio * 100.0) << "%)" << std::endl;
    std::cout << "Negative paragraphs: " << negativeCount << " (" 
              << std::fixed << std::setprecision(1) << ((1.0 - positiveRatio) * 100.0) << "%)" << std::endl;
    
    // Calculate average confidence
    double totalConfidence = 0.0;
    
    const auto sumConfidence = [&](auto& self, size_t index) -> void {
        // Base case: all confidences summed
        if (index >= results.size()) {
            return;
        }
        
        // Add this confidence
        totalConfidence += results[index].confidence;
        
        // Recursively sum next confidence
        self(self, index + 1);
    };
    
    sumConfidence(sumConfidence, 0);
    
    double avgConfidence = results.empty() ? 0.0 : totalConfidence / results.size();
    std::cout << "Average confidence: " << std::fixed << std::setprecision(2) << avgConfidence << std::endl;
    
    // Find highest and lowest confidence recursively
    const auto findExtremesRecursive = [&results](auto& self, size_t index, 
                                            size_t& highestIdx, size_t& lowestIdx,
                                            double highestConf, double lowestConf) -> void {
        // Base case: all results checked
        if (index >= results.size()) {
            return;
        }
        
        double conf = results[index].confidence;
        
        // Update highest confidence if needed
        if (conf > highestConf) {
            highestConf = conf;
            highestIdx = index;
        }
        
        // Update lowest confidence if needed
        if (conf < lowestConf) {
            lowestConf = conf;
            lowestIdx = index;
        }
        
        // Recursively check next result
        self(self, index + 1, highestIdx, lowestIdx, highestConf, lowestConf);
    };
    
    size_t highestIdx = 0;
    size_t lowestIdx = 0;
    
    if (!results.empty()) {
        findExtremesRecursive(findExtremesRecursive, 0, highestIdx, lowestIdx, 
                           results[0].confidence, results[0].confidence);
        
        // Print most confident results
        std::cout << "\nMost confident prediction:" << std::endl;
        std::cout << "Sentiment: " << results[highestIdx].sentiment << std::endl;
        std::cout << "Confidence: " << results[highestIdx].confidence << std::endl;
        std::cout << "Text: " << results[highestIdx].text.substr(0, 100) 
                  << (results[highestIdx].text.size() > 100 ? "..." : "") << std::endl;
        
        // Print least confident results
        std::cout << "\nLeast confident prediction:" << std::endl;
        std::cout << "Sentiment: " << results[lowestIdx].sentiment << std::endl;
        std::cout << "Confidence: " << results[lowestIdx].confidence << std::endl;
        std::cout << "Text: " << results[lowestIdx].text.substr(0, 100)
                  << (results[lowestIdx].text.size() > 100 ? "..." : "") << std::endl;
    }
    
    // Save report to file
    std::string reportPath = std::filesystem::path(filePath).stem().string() + "_sentiment_report.txt";
    std::ofstream reportFile(reportPath);
    
    if (reportFile.is_open()) {
        reportFile << "Sentiment Analysis Report for: " << std::filesystem::path(filePath).filename() << std::endl;
        reportFile << "===================================================" << std::endl;
        reportFile << "Total paragraphs analyzed: " << results.size() << std::endl;
        reportFile << "Positive paragraphs: " << positiveCount << " (" 
                   << std::fixed << std::setprecision(1) << (positiveRatio * 100.0) << "%)" << std::endl;
        reportFile << "Negative paragraphs: " << negativeCount << " (" 
                   << std::fixed << std::setprecision(1) << ((1.0 - positiveRatio) * 100.0) << "%)" << std::endl;
        reportFile << "Average confidence: " << std::fixed << std::setprecision(2) << avgConfidence << std::endl;
        
        // Add detailed results recursively
        const auto writeDetailedResults = [&reportFile](auto& self, const auto& results, size_t index) -> void {
            // Base case: all results written
            if (index >= results.size()) {
                return;
            }
            
            const auto& result = results[index];
            
            reportFile << "\n----- Paragraph " << (index + 1) << " -----" << std::endl;
            reportFile << "Sentiment: " << result.sentiment << std::endl;
            reportFile << "Confidence: " << result.confidence << std::endl;
            reportFile << "Text: " << result.text << std::endl;
            
            // Recursively write next result
            self(self, results, index + 1);
        };
        
        reportFile << "\nDetailed Analysis:" << std::endl;
        writeDetailedResults(writeDetailedResults, results, 0);
        
        reportFile.close();
        std::cout << "\nReport saved to: " << reportPath << std::endl;
    } else {
        std::cerr << "Error: Could not save report file" << std::endl;
    }
    
    return true;
}

/**
 * Main file analyzer function that processes command-line arguments.
 * 
 * This function demonstrates command-line argument handling and
 * initializes the sentiment analysis components before processing files.
 * 
 * @param argc Number of command-line arguments
 * @param argv Array of command-line argument strings
 * @return Exit status code
 */
int main(int argc, char* argv[]) {
    // Check if file path is provided
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <file_path> [file_path2 ...]" << std::endl;
        std::cout << "Example: " << argv[0] << " sample.txt another_file.txt" << std::endl;
        return 1;
    }
    
    try {
        // Initialize sentiment analyzer
        nlp::SentimentAnalyzer analyzer;
        
        // Create and train model with sample data
        std::cout << "Initializing sentiment analyzer with sample data..." << std::endl;
        
        // Sample training data
        std::vector<std::pair<int, std::string>> sampleData = {
            {4, "This product is amazing! I love it so much."},
            {4, "Great experience, would definitely recommend."},
            {4, "The service was excellent and staff very friendly."},
            {4, "I'm completely satisfied with my purchase."},
            {4, "Best customer support I've ever encountered."},
            {0, "Terrible product, broke after first use."},
            {0, "Very disappointed with the quality."},
            {0, "Customer service was awful and unhelpful."},
            {0, "Would not recommend to anyone."},
            {0, "Worst purchase I've ever made."}
        };
        
        // Create dataset
        nlp::SentimentDataset dataset;
        dataset.data = sampleData;
        
        // Preprocess data - Using std::move to avoid copying the dataset
        dataset = analyzer.preprocessData(std::move(dataset));
        
        // Extract features
        auto [features, labels] = analyzer.extractFeatures(dataset);
        dataset.features = features;
        dataset.labels = labels;
        
        // Train model
        std::cout << "Training sentiment model..." << std::endl;
        auto model = analyzer.trainModel(features, labels);
        
        // Create vectorizer
        nlp::TfidfVectorizer vectorizer(true, 0.5, 100);
        vectorizer.fitTransform(dataset.cleanedTexts);
        
        // Process each file recursively
        const auto processFiles = [&](auto& self, int index, int argc, char* argv[]) -> bool {
            // Base case: all files processed
            if (index >= argc) {
                return true;
            }
            
            // Process this file
            std::string filePath = argv[index];
            bool success = analyzeSingleFile(analyzer, *model, vectorizer, filePath);
            
            if (!success) {
                std::cerr << "Failed to analyze file: " << filePath << std::endl;
            }
            
            // Add separator between files
            if (index < argc - 1) {
                std::cout << "\n---------------------------------------------\n" << std::endl;
            }
            
            // Recursively process next file
            return self(self, index + 1, argc, argv);
        };
        
        processFiles(processFiles, 1, argc, argv);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

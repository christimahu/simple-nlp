/**
 * @file word_cloud.cpp
 * @brief Example application for generating ASCII word clouds
 * 
 * This example demonstrates how to use the AsciiWordCloud class
 * to visualize text data. It shows various customization options
 * and different ways to generate word clouds for text analysis.
 */

#include "ascii_word_cloud.h"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>

/**
 * Function to demonstrate various word cloud features.
 * 
 * This function shows how to generate different types of
 * word clouds, customize their appearance, and save them to files.
 * It demonstrates both positive and negative sentiment visualizations.
 */
void wordCloudDemo() {
    std::cout << "==== ASCII Word Cloud Demo ====" << std::endl;
    
    // Create sample text data for positive sentiment
    std::vector<std::string> positiveTexts = {
        "amazing fantastic wonderful excellent great awesome",
        "amazing excellent helpful responsive service support",
        "fantastic product quality reliable durable comfortable",
        "wonderful experience incredible delightful exceeded expectations",
        "great value worth every penny excellent purchase"
    };
    
    // Create sample text data for negative sentiment
    std::vector<std::string> negativeTexts = {
        "terrible horrible awful disappointing frustrating",
        "terrible quality broken defective waste money",
        "horrible service rude unhelpful slow unresponsive",
        "awful experience disappointed regret purchase returning",
        "frustrating difficult confusing complicated inconvenient"
    };
    
    // Create an instance of the AsciiWordCloud
    nlp::AsciiWordCloud wordCloud;
    
    // Basic word cloud generation
    std::cout << "\nGenerating basic positive word cloud:" << std::endl;
    std::string positiveCloud = wordCloud.generateWordCloud(positiveTexts, 10, 60, 10, true);
    std::cout << positiveCloud << std::endl;
    
    // Display negative word cloud with color
    std::cout << "\nDisplaying negative word cloud with color:" << std::endl;
    wordCloud.displayWordCloud(negativeTexts, 10, 60, 10, false);
    
    // Custom word cloud configuration
    nlp::AsciiWordCloud::CloudConfig customConfig;
    customConfig.maxWords = 8;
    customConfig.width = 50;
    customConfig.height = 8;
    customConfig.useColor = true;
    customConfig.useBars = true;
    customConfig.showFrequencies = true;
    
    std::cout << "\nGenerating custom configured word cloud:" << std::endl;
    std::string customCloud = wordCloud.generateCustomCloud(positiveTexts, customConfig, true);
    std::cout << customCloud << std::endl;
    
    // Save word cloud to file
    std::string filename = "word_cloud.txt";
    std::ofstream outFile(filename);
    
    if (outFile.is_open()) {
        // Create a larger word cloud for the file
        customConfig.width = 80;
        customConfig.height = 20;
        customConfig.maxWords = 15;
        
        std::string fileCloud = wordCloud.generateCustomCloud(positiveTexts, customConfig, true);
        outFile << fileCloud;
        outFile.close();
        
        std::cout << "Word cloud saved to " << filename << std::endl;
    } else {
        std::cerr << "Unable to save word cloud to file" << std::endl;
    }
    
    // Demonstrate combining different texts
    std::cout << "\nCombining text sources for word cloud:" << std::endl;
    
    // Combine texts recursively
    std::vector<std::string> combinedTexts;
    
    const auto combineTexts = [&](auto& self, const auto& texts, size_t index) -> void {
        // Base case: all texts processed
        if (index >= texts.size()) {
            return;
        }
        
        // Add this text
        combinedTexts.push_back(texts[index]);
        
        // Recursively process next text
        self(self, texts, index + 1);
    };
    
    // Process positive texts
    combineTexts(combineTexts, positiveTexts, 0);
    
    // Process negative texts
    combineTexts(combineTexts, negativeTexts, 0);
    
    // Generate combined cloud
    customConfig.width = 70;
    customConfig.height = 15;
    customConfig.maxWords = 12;
    customConfig.useColor = false; // No color for combined cloud
    
    std::string combinedCloud = wordCloud.generateCustomCloud(combinedTexts, customConfig, true);
    std::cout << combinedCloud << std::endl;
    
    std::cout << "==== Word Cloud Demo Complete ====" << std::endl;
}

/**
 * Main function for the word cloud example application.
 * 
 * @return Exit status code (0 for success, 1 for error)
 */
int main() {
    try {
        wordCloudDemo();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

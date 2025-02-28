#include "sentiment_analysis.h"
#include <algorithm>
#include <iostream>
#include <sstream>
#include <random>
#include <iomanip>
#include <cmath>
#include <fstream>

namespace nlp {

std::string AsciiWordCloud::generateWordCloud(
    const std::vector<std::string>& texts, 
    size_t maxWords, 
    size_t width, 
    size_t height,
    bool isPositive) {
    
    // Count word frequencies
    auto wordFreqs = countWordFrequencies(texts);
    
    // Get top words
    auto topWords = getTopWords(wordFreqs, maxWords);
    
    if (topWords.empty()) {
        return "No words found to create a word cloud.";
    }
    
    // Find maximum frequency
    int maxFreq = topWords[0].second;
    
    // Create cloud layout
    std::stringstream cloud;
    
    // Add header
    std::string sentiment = isPositive ? "POSITIVE" : "NEGATIVE";
    cloud << std::string(width/2 - 10, '*') << " " << sentiment << " WORD CLOUD " << std::string(width/2 - 10, '*') << "\n\n";
    
    // Random number generator for positioning
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> widthDist(0, std::max(1, static_cast<int>(width - 20)));
    
    // Initialize cloud canvas
    std::vector<std::string> canvas(height, std::string(width, ' '));
    
    // Place top words in the canvas
    for (const auto& [word, freq] : topWords) {
        std::string formattedWord = formatWord(word, freq, maxFreq, isPositive);
        
        // Try to find a position (simple approach)
        int attempts = 0;
        bool placed = false;
        
        while (!placed && attempts < 10) {
            int row = attempts % height;
            int col = widthDist(gen);
            
            // Check if there's space
            if (canvas[row].substr(col, formattedWord.length()).find_first_not_of(' ') == std::string::npos) {
                // Copy the formatted word into the canvas
                for (size_t i = 0; i < formattedWord.length() && col + i < width; ++i) {
                    canvas[row][col + i] = formattedWord[i];
                }
                placed = true;
            }
            attempts++;
        }
        
        // If we couldn't place it nicely, just append it
        if (!placed) {
            cloud << formattedWord << " ";
        }
    }
    
    // Add canvas to output
    for (const auto& line : canvas) {
        cloud << line << "\n";
    }
    
    // Add legend
    cloud << "\n" << std::string(width, '-') << "\n";
    cloud << "Word size represents frequency. ";
    cloud << "Based on " << texts.size() << " texts with " << (isPositive ? "positive" : "negative") << " sentiment.\n";
    
    return cloud.str();
}

void AsciiWordCloud::displayWordCloud(
    const std::vector<std::string>& texts, 
    size_t maxWords, 
    size_t width, 
    size_t height,
    bool isPositive) {
    
    // Count word frequencies
    auto wordFreqs = countWordFrequencies(texts);
    
    // Get top words
    auto topWords = getTopWords(wordFreqs, maxWords);
    
    if (topWords.empty()) {
        std::cout << "No words found to create a word cloud." << std::endl;
        return;
    }
    
    // Find maximum frequency
    int maxFreq = topWords[0].second;
    
    // Add header
    std::string sentiment = isPositive ? "POSITIVE" : "NEGATIVE";
    std::string headerColor = isPositive ? "\033[1;32m" : "\033[1;31m"; // Green for positive, Red for negative
    
    std::cout << headerColor << std::string(width/2 - 10, '*') << " " << sentiment << " WORD CLOUD " 
              << std::string(width/2 - 10, '*') << resetColor() << "\n\n";
    
    // Print top words with appropriate styling
    for (const auto& [word, freq] : topWords) {
        // Determine size and color based on frequency
        double normFreq = static_cast<double>(freq) / maxFreq;
        
        // Get color based on sentiment and frequency
        std::string colorCode = getColorCode(freq, maxFreq, isPositive);
        
        // Print word with appropriate styling
        std::cout << colorCode << std::setw(20) << std::left << word;
        
        // Print a bar showing frequency
        int barLength = static_cast<int>(30 * normFreq);
        for (int i = 0; i < barLength; ++i) {
            std::cout << "█";
        }
        
        // Print frequency count
        std::cout << " " << freq << resetColor() << std::endl;
    }
    
    // Print legend
    std::cout << std::string(width, '-') << std::endl;
    std::cout << "Word frequency visualization for " << texts.size() << " texts with " 
              << (isPositive ? "positive" : "negative") << " sentiment." << std::endl;
    
    // Add some whitespace
    std::cout << std::endl << std::endl;
}

std::unordered_map<std::string, int> AsciiWordCloud::countWordFrequencies(
    const std::vector<std::string>& texts) {
    
    std::unordered_map<std::string, int> wordFreqs;
    
    for (const auto& text : texts) {
        std::istringstream iss(text);
        std::string word;
        
        while (iss >> word) {
            // Only count words at least 3 characters long
            if (word.length() >= 3) {
                wordFreqs[word]++;
            }
        }
    }
    
    return wordFreqs;
}

std::vector<std::pair<std::string, int>> AsciiWordCloud::getTopWords(
    const std::unordered_map<std::string, int>& wordFreqs, 
    size_t maxWords) {
    
    // Convert map to vector for sorting
    std::vector<std::pair<std::string, int>> words;
    words.reserve(wordFreqs.size());
    
    for (const auto& [word, freq] : wordFreqs) {
        words.push_back({word, freq});
    }
    
    // Sort by frequency (descending)
    std::sort(words.begin(), words.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Take top N words
    if (words.size() > maxWords) {
        words.resize(maxWords);
    }
    
    return words;
}

std::string AsciiWordCloud::formatWord(
    const std::string& word, 
    int freq, 
    int maxFreq,
    bool isPositive) {
    
    // Calculate relative font size (1-5)
    double normFreq = static_cast<double>(freq) / maxFreq;
    int fontSize = 1 + static_cast<int>(normFreq * 4);
    
    // Symbols to use for emphasis
    std::vector<std::string> symbols = {".", "~", "*", "%", "#"};
    std::string symbol = symbols[fontSize - 1];
    
    // Format based on font size
    std::stringstream formatted;
    formatted << symbol;
    
    // For larger words, add more emphasis
    if (fontSize >= 3) {
        formatted << symbol;
    }
    
    formatted << word;
    
    if (fontSize >= 3) {
        formatted << symbol;
    }
    
    formatted << symbol;
    
    return formatted.str();
}

std::string AsciiWordCloud::getColorCode(int freq, int maxFreq, bool isPositive) {
    double normFreq = static_cast<double>(freq) / maxFreq;
    
    // Color intensity based on frequency (0-5)
    int intensity = static_cast<int>(normFreq * 5);
    
    if (isPositive) {
        // Green shades for positive sentiment
        switch (intensity) {
            case 0: return "\033[32m";       // Dark green
            case 1: return "\033[1;32m";     // Bold green
            case 2: return "\033[1;92m";     // Bold light green
            case 3: return "\033[1;92m";     // Bold light green
            case 4: return "\033[1;92;4m";   // Bold light green, underlined
            default: return "\033[1;92;4m";  // Bold light green, underlined
        }
    } else {
        // Red shades for negative sentiment
        switch (intensity) {
            case 0: return "\033[31m";       // Dark red
            case 1: return "\033[1;31m";     // Bold red
            case 2: return "\033[1;91m";     // Bold light red
            case 3: return "\033[1;91m";     // Bold light red
            case 4: return "\033[1;91;4m";   // Bold light red, underlined
            default: return "\033[1;91;4m";  // Bold light red, underlined
        }
    }
}

std::string AsciiWordCloud::resetColor() {
    return "\033[0m";
}

} // namespace nlp

/**
 * @file ascii_word_cloud.cpp
 * @brief Implementation of the AsciiWordCloud class
 * 
 * This file contains the implementation of methods for generating ASCII-based
 * word clouds from text data. It demonstrates functional programming using
 * recursion instead of traditional loops, and includes detailed explanations
 * of the algorithms and design choices.
 */

#include "sentiment_analysis.h"
#include <algorithm>
#include <iostream>
#include <sstream>
#include <random>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <functional>

namespace nlp {

/**
 * Generates a word cloud from a collection of texts with a specified sentiment.
 * 
 * This function demonstrates a functional approach to data processing, using
 * helper functions for each step of the word cloud generation process. It handles
 * the full pipeline from word frequency counting to cloud layout and formatting.
 * 
 * @param texts Collection of texts with the specified sentiment
 * @param maxWords Maximum number of words to include in the cloud
 * @param width Width of the ASCII display
 * @param height Height of the ASCII display
 * @param isPositive Whether this is a positive sentiment cloud (affects colors)
 * @return Formatted ASCII word cloud as a string
 */
std::string AsciiWordCloud::generateWordCloud(
    std::span<const std::string> texts, 
    size_t maxWords, 
    size_t width, 
    size_t height,
    bool isPositive) {
    
    // Count word frequencies using our recursive approach
    auto wordFreqs = countWordFrequencies(texts);
    
    // Get top words by frequency
    auto topWords = getTopWords(wordFreqs, maxWords);
    
    if (topWords.empty()) {
        return "No words found to create a word cloud.";
    }
    
    // Find maximum frequency
    int maxFreq = topWords[0].second;
    
    // Create cloud layout
    std::stringstream cloud;
    
    // Add header using a functional approach
    const auto addHeader = [isPositive, width](std::stringstream& ss) -> void {
        std::string sentiment = isPositive ? "POSITIVE" : "NEGATIVE";
        ss << std::string(width/2 - 10, '*') << " " << sentiment 
           << " WORD CLOUD " << std::string(width/2 - 10, '*') << "\n\n";
    };
    
    addHeader(cloud);
    
    // Random number generator for positioning
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> widthDist(0, std::max(1, static_cast<int>(width - 20)));
    
    // Initialize cloud canvas
    std::vector<std::string> canvas(height, std::string(width, ' '));
    
    // Place top words in the canvas using a recursive approach
    const auto placeWords = [&](auto& self, const auto& words, size_t index) -> void {
        // Base case: all words placed
        if (index >= words.size()) {
            return;
        }
        
        const auto& [word, freq] = words[index];
        std::string formattedWord = formatWord(word, freq, maxFreq, isPositive);
        
        // Try to find a position (recursive approach)
        const auto tryPlacement = [&](auto& self, int attempt) -> bool {
            // Base case: too many attempts
            if (attempt >= 10) {
                return false;
            }
            
            int row = attempt % height;
            int col = widthDist(gen);
            
            // Check if there's space by examining each character position
            const auto checkSpace = [&](auto& self, size_t pos) -> bool {
                // Base case: all positions checked and all are spaces
                if (pos >= formattedWord.length() || col + pos >= width) {
                    return true;
                }
                
                // If non-space character found, position is not valid
                if (canvas[row][col + pos] != ' ') {
                    return false;
                }
                
                // Recursively check next position
                return self(self, pos + 1);
            };
            
            bool hasSpace = checkSpace(checkSpace, 0);
            
            if (hasSpace) {
                // Place the word by copying each character
                const auto placeWord = [&](auto& self, size_t pos) -> void {
                    // Base case: all characters placed or out of bounds
                    if (pos >= formattedWord.length() || col + pos >= width) {
                        return;
                    }
                    
                    // Set character and recursively place next character
                    canvas[row][col + pos] = formattedWord[pos];
                    self(self, pos + 1);
                };
                
                placeWord(placeWord, 0);
                return true;
            } else {
                // Recursively try next placement attempt
                return self(self, attempt + 1);
            }
        };
        
        bool placed = tryPlacement(tryPlacement, 0);
        
        // If we couldn't place it nicely, just append it
        if (!placed) {
            cloud << formattedWord << " ";
        }
        
        // Recursively place next word
        self(self, words, index + 1);
    };
    
    placeWords(placeWords, topWords, 0);
    
    // Add canvas to output using a recursive approach
    const auto addCanvas = [&](auto& self, size_t lineIndex) -> void {
        // Base case: all lines added
        if (lineIndex >= canvas.size()) {
            return;
        }
        
        // Add this line and recursively add the next
        cloud << canvas[lineIndex] << "\n";
        self(self, lineIndex + 1);
    };
    
    addCanvas(addCanvas, 0);
    
    // Add legend
    cloud << "\n" << std::string(width, '-') << "\n";
    cloud << "Word size represents frequency. ";
    cloud << "Based on " << texts.size() << " texts with " 
          << (isPositive ? "positive" : "negative") << " sentiment.\n";
    
    return cloud.str();
}

/**
 * Displays a word cloud to the console with ANSI color formatting.
 * 
 * This function demonstrates a visual representation of text data,
 * using color and formatting to convey frequency information. It uses
 * recursion to process and display words.
 * 
 * @param texts Collection of texts with the specified sentiment
 * @param maxWords Maximum number of words to include in the cloud
 * @param width Width of the ASCII display
 * @param height Height of the ASCII display
 * @param isPositive Whether this is a positive sentiment cloud (affects colors)
 */
void AsciiWordCloud::displayWordCloud(
    std::span<const std::string> texts, 
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
    
    // Add header using a functional approach
    const auto printHeader = [isPositive, width]() -> void {
        std::string sentiment = isPositive ? "POSITIVE" : "NEGATIVE";
        std::string headerColor = isPositive ? "\033[1;32m" : "\033[1;31m"; // Green for positive, Red for negative
        
        std::cout << headerColor << std::string(width/2 - 10, '*') << " " << sentiment << " WORD CLOUD " 
                  << std::string(width/2 - 10, '*') << resetColor() << "\n\n";
    };
    
    printHeader();
    
    // Print words with a recursive approach
    const auto printWords = [maxFreq, isPositive](auto& self, const auto& words, size_t index) -> void {
        // Base case: all words printed
        if (index >= words.size()) {
            return;
        }
        
        const auto& [word, freq] = words[index];
        
        // Determine size and color based on frequency
        double normFreq = static_cast<double>(freq) / maxFreq;
        
        // Get color based on sentiment and frequency
        std::string colorCode = getColorCode(freq, maxFreq, isPositive);
        
        // Print word with appropriate styling
        std::cout << colorCode << std::setw(20) << std::left << word;
        
        // Print a bar showing frequency using recursion
        const auto printBar = [](auto& self, int current, int total) -> void {
            // Base case: full bar printed
            if (current >= total) {
                return;
            }
            
            // Print bar segment and recursively print next segment
            std::cout << "█";
            self(self, current + 1, total);
        };
        
        int barLength = static_cast<int>(30 * normFreq);
        printBar(printBar, 0, barLength);
        
        // Print frequency count
        std::cout << " " << freq << resetColor() << std::endl;
        
        // Recursively print next word
        self(self, words, index + 1);
    };
    
    printWords(printWords, topWords, 0);
    
    // Print legend
    std::cout << std::string(width, '-') << std::endl;
    std::cout << "Word frequency visualization for " << texts.size() << " texts with " 
              << (isPositive ? "positive" : "negative") << " sentiment." << std::endl;
    
    // Add some whitespace
    std::cout << std::endl << std::endl;
}

/**
 * Generates a word cloud with custom configuration options.
 * 
 * This function extends the basic word cloud generation with
 * additional customization options, demonstrating how to create
 * flexible, configurable components.
 * 
 * @param texts Collection of texts
 * @param config Configuration options
 * @param isPositive Whether this is a positive sentiment cloud
 * @return Generated word cloud as a string
 */
std::string AsciiWordCloud::generateCustomCloud(
    std::span<const std::string> texts,
    const CloudConfig& config,
    bool isPositive) {
    
    // Count word frequencies
    auto wordFreqs = countWordFrequencies(texts);
    
    // Get top words
    auto topWords = getTopWords(wordFreqs, config.maxWords);
    
    if (topWords.empty()) {
        return "No words found to create a word cloud.";
    }
    
    // Find maximum frequency
    int maxFreq = topWords[0].second;
    
    // Create cloud layout
    std::stringstream cloud;
    
    // Add header
    std::string sentiment = isPositive ? "POSITIVE" : "NEGATIVE";
    cloud << std::string(config.width/2 - 10, '*') << " " << sentiment 
          << " WORD CLOUD " << std::string(config.width/2 - 10, '*') << "\n\n";
    
    // Print words based on configuration using a recursive approach
    const auto printWords = [&](auto& self, size_t index) -> void {
        // Base case: all words printed
        if (index >= topWords.size()) {
            return;
        }
        
        const auto& [word, freq] = topWords[index];
        double normFreq = static_cast<double>(freq) / maxFreq;
        
        // Apply formatting based on configuration
        if (config.useColor) {
            cloud << getColorCode(freq, maxFreq, isPositive);
        }
        
        cloud << std::setw(20) << std::left << word;
        
        // Add frequency bar if enabled
        if (config.useBars) {
            int barLength = static_cast<int>(30 * normFreq);
            
            // Print bar using recursion
            const auto printBar = [&](auto& self, int current) -> void {
                // Base case: full bar printed
                if (current >= barLength) {
                    return;
                }
                
                // Print bar segment and recursively print next segment
                cloud << "█";
                self(self, current + 1);
            };
            
            printBar(printBar, 0);
        }
        
        // Add frequency count if enabled
        if (config.showFrequencies) {
            cloud << " " << freq;
        }
        
        if (config.useColor) {
            cloud << resetColor();
        }
        
        cloud << "\n";
        
        // Recursively print next word
        self(self, index + 1);
    };
    
    printWords(printWords, 0);
    
    // Add footer
    cloud << std::string(config.width, '-') << "\n";
    cloud << "Word cloud based on " << texts.size() << " texts.\n";
    
    return cloud.str();
}

/**
 * Counts word frequencies in a collection of texts.
 * 
 * This function demonstrates a recursive approach to text processing,
 * counting the frequency of each word across multiple texts. This is
 * a fundamental operation for generating word clouds.
 * 
 * @param texts Collection of texts
 * @return Map of words to their frequencies
 */
std::unordered_map<std::string, int> AsciiWordCloud::countWordFrequencies(
    std::span<const std::string> texts) {
    
    std::unordered_map<std::string, int> wordFreqs;
    
    // Process each text recursively
    const auto processTexts = [&](auto& self, size_t textIndex) -> void {
        // Base case: all texts processed
        if (textIndex >= texts.size()) {
            return;
        }
        
        // Process words in this text
        std::istringstream iss(texts[textIndex]);
        
        // Process each word recursively
        const auto processWords = [&](auto& self) -> void {
            std::string word;
            
            // Base case: no more words to extract
            if (!(iss >> word)) {
                return;
            }
            
            // Only count words at least 3 characters long
            if (word.length() >= 3) {
                wordFreqs[word]++;
            }
            
            // Recursively process next word
            self(self);
        };
        
        processWords(processWords);
        
        // Recursively process next text
        self(self, textIndex + 1);
    };
    
    processTexts(processTexts, 0);
    
    return wordFreqs;
}

/**
 * Gets the top N words by frequency.
 * 
 * This function demonstrates sorting and filtering operations in
 * a functional style, using recursion for data transformation.
 * 
 * @param wordFreqs Word frequency map
 * @param maxWords Maximum number of words to include
 * @return Vector of (word, frequency) pairs, sorted by frequency
 */
std::vector<std::pair<std::string, int>> AsciiWordCloud::getTopWords(
    const std::unordered_map<std::string, int>& wordFreqs, 
    size_t maxWords) {
    
    // Convert map to vector for sorting
    std::vector<std::pair<std::string, int>> words;
    words.reserve(wordFreqs.size());
    
    // Transfer words recursively
    const auto transferWords = [&](auto& self, auto it) -> void {
        // Base case: all words transferred
        if (it == wordFreqs.end()) {
            return;
        }
        
        // Add this word-frequency pair
        words.push_back(*it);
        
        // Recursively transfer next word
        self(self, std::next(it));
    };
    
    transferWords(transferWords, wordFreqs.begin());
    
    // Sort by frequency (descending) using a custom comparator
    std::sort(words.begin(), words.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Limit to top N words
    if (words.size() > maxWords) {
        words.resize(maxWords);
    }
    
    return words;
}

/**
 * Formats a word with ASCII art based on its frequency.
 * 
 * This function demonstrates string manipulation and visual formatting
 * in a functional style. It applies emphasis and decorations to words
 * based on their frequency in the corpus.
 * 
 * @param word The word to format
 * @param freq The word's frequency
 * @param maxFreq The maximum frequency in the set
 * @param isPositive Whether this is for a positive sentiment cloud
 * @return Formatted word with ASCII art
 */
std::string AsciiWordCloud::formatWord(
    std::string_view word, 
    int freq, 
    int maxFreq,
    bool isPositive) {
    
    // Calculate relative font size (1-5)
    double normFreq = static_cast<double>(freq) / maxFreq;
    int fontSize = 1 + static_cast<int>(normFreq * 4);
    
    // Symbols to use for emphasis
    std::vector<std::string> symbols = {".", "~", "*", "%", "#"};
    std::string symbol = symbols[fontSize - 1];
    
    // Format based on font size using a functional approach
    std::stringstream formatted;
    
    // Add emphasis symbols recursively
    const auto addEmphasis = [&](auto& self, int level) -> void {
        // Base case: all emphasis added
        if (level <= 0) {
            return;
        }
        
        // Add emphasis and recursively continue
        formatted << symbol;
        self(self, level - 1);
    };
    
    // Add prefix emphasis
    addEmphasis(addEmphasis, fontSize >= 3 ? 2 : 1);
    
    // Add the word itself
    formatted << word;
    
    // Add suffix emphasis
    addEmphasis(addEmphasis, fontSize >= 3 ? 2 : 1);
    
    return formatted.str();
}

/**
 * Gets ANSI color code for a word based on its frequency and sentiment.
 * 
 * This function demonstrates how to provide visual feedback using terminal
 * colors, with selection based on multiple criteria (frequency and sentiment).
 * 
 * @param freq Word frequency
 * @param maxFreq Maximum frequency in set
 * @param isPositive Whether this is for a positive sentiment cloud
 * @return ANSI color code
 */
std::string AsciiWordCloud::getColorCode(int freq, int maxFreq, bool isPositive) {
    double normFreq = static_cast<double>(freq) / maxFreq;
    
    // Color intensity based on frequency (0-5)
    int intensity = static_cast<int>(normFreq * 5);
    
    // A purely functional approach would use a map or recursive function
    // to select the color, but a simple switch is clearer here
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

/**
 * Resets ANSI color formatting.
 * 
 * A simple utility function to reset terminal color settings.
 * 
 * @return ANSI reset code
 */
std::string AsciiWordCloud::resetColor() {
    return "\033[0m";
}

} // namespace nlp

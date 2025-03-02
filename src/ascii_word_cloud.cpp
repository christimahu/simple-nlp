/**
 * @file ascii_word_cloud.cpp
 * @brief Implements ASCII art word clouds for text visualization.
 * 
 * This file provides the implementation for generating ASCII-based word clouds
 * that visualize the frequency of words in a collection of texts. It includes
 * different visualization styles, color options, and customization features.
 */

#include "ascii_word_cloud.h"
#include <algorithm>
#include <iostream>
#include <sstream>
#include <random>
#include <iomanip>
#include <cmath>
#include <cctype>

namespace nlp {

/**
 * @brief Generates an ASCII word cloud from a collection of texts.
 * 
 * This method creates a simple ASCII visualization of word frequencies,
 * with more frequent words appearing larger or more prominently.
 * 
 * @param texts Collection of text documents to analyze.
 * @param maxWords Maximum number of words to include.
 * @param width Width of the word cloud in characters.
 * @param height Height of the word cloud in characters.
 * @param isPositive Whether to use positive sentiment styling.
 * @return A string containing the ASCII word cloud.
 */
std::string AsciiWordCloud::generateWordCloud(
    const std::vector<std::string>& texts,
    size_t maxWords, 
    size_t width, 
    size_t height,
    bool isPositive) {
    
    // Use the CloudConfig struct for configuration
    CloudConfig config;
    config.maxWords = maxWords;
    config.width = width;
    config.height = height;
    config.useColor = true;
    
    return generateCustomCloud(texts, config, isPositive);
}

/**
 * @brief Generates a customized ASCII word cloud.
 * 
 * This method provides more control over the word cloud generation
 * through a configuration struct that allows setting various parameters.
 * 
 * @param texts Collection of text documents to analyze.
 * @param config Configuration options for the word cloud.
 * @param isPositive Whether to use positive sentiment styling.
 * @return A string containing the customized ASCII word cloud.
 */
std::string AsciiWordCloud::generateCustomCloud(
    const std::vector<std::string>& texts,
    const CloudConfig& config,
    bool isPositive) {
    
    // Count word frequencies
    auto wordFreqs = countWordFrequencies(texts);
    
    // Get top words
    auto topWords = getTopWords(wordFreqs, config.maxWords);
    
    if (topWords.empty()) {
        return "No words found to create a word cloud.";
    }
    
    // Find the maximum frequency for scaling
    int maxFreq = topWords[0].second;
    
    // Prepare the output stream
    std::ostringstream cloud;
    
    // Add a title
    cloud << "Word Frequency Visualization for " << texts.size() << " texts";
    cloud << (isPositive ? " with positive sentiment." : " with negative sentiment.") << "\n\n";
    
    // Two possible visualization styles
    if (config.useBars) {
        // Bar chart style
        for (const auto& [word, freq] : topWords) {
            // Calculate bar length scaled to width
            int barLength = static_cast<int>(
                (static_cast<double>(freq) / maxFreq) * (config.width - 20)
            );
            
            // Format the word and frequency
            cloud << std::left << std::setw(15) << word;
            cloud << " ";
            
            // Add the bar
            if (config.useColor) {
                cloud << getColorCode(freq, maxFreq, isPositive);
            }
            
            for (int i = 0; i < barLength; ++i) {
                cloud << "█";
            }
            
            if (config.useColor) {
                cloud << resetColor();
            }
            
            // Show frequency if requested
            if (config.showFrequencies) {
                cloud << " (" << freq << ")";
            }
            
            cloud << "\n";
        }
    } else {
        // Free-form cloud style
        cloud << "Top Words:\n";
        for (const auto& [word, freq] : topWords) {
            cloud << formatWord(word, freq, maxFreq, isPositive);
            cloud << " ";
            
            // Show frequency if requested
            if (config.showFrequencies) {
                cloud << "(" << freq << ")";
            }
            
            cloud << "\n";
        }
    }
    
    return cloud.str();
}

/**
 * @brief Displays a word cloud directly in the console.
 * 
 * This method outputs the word cloud to the console with
 * ANSI color formatting for visual enhancement.
 * 
 * @param texts Collection of text documents to analyze.
 * @param maxWords Maximum number of words to include.
 * @param width Width of the word cloud in characters.
 * @param height Height of the word cloud in characters.
 * @param isPositive Whether to use positive sentiment styling.
 */
void AsciiWordCloud::displayWordCloud(
    const std::vector<std::string>& texts,
    size_t maxWords, 
    size_t width, 
    size_t height,
    bool isPositive) {
    
    // Generate the word cloud
    std::string cloudText = generateWordCloud(texts, maxWords, width, height, isPositive);
    
    // Display directly to console
    std::cout << cloudText << std::endl;
}

/**
 * @brief Counts the frequency of words in a collection of texts.
 * 
 * This method tokenizes the input texts and creates a frequency map
 * of words, ignoring very short words (less than 3 characters).
 * 
 * @param texts Collection of text documents to analyze.
 * @return Map of words to their frequency counts.
 */
std::unordered_map<std::string, int> AsciiWordCloud::countWordFrequencies(
    const std::vector<std::string>& texts) {
    
    std::unordered_map<std::string, int> wordFreqs;
    
    // Process each text
    for (const auto& text : texts) {
        std::istringstream iss{text};
        std::string word;
        
        // Count each word
        while (iss >> word) {
            // Filter out very short words
            if (word.length() >= 3) {
                // Convert to lowercase for consistent counting
                std::transform(word.begin(), word.end(), word.begin(),
                              [](unsigned char c) { return std::tolower(c); });
                
                // Increment frequency
                wordFreqs[word]++;
            }
        }
    }
    
    return wordFreqs;
}

/**
 * @brief Extracts the most frequent words from a frequency map.
 * 
 * This method sorts words by frequency and returns the top N words.
 * 
 * @param wordFreqs Map of words to their frequencies.
 * @param maxWords Maximum number of words to return.
 * @return Vector of (word, frequency) pairs sorted by frequency.
 */
std::vector<std::pair<std::string, int>> AsciiWordCloud::getTopWords(
    const std::unordered_map<std::string, int>& wordFreqs, 
    size_t maxWords) {
    
    // Convert map to vector for sorting
    std::vector<std::pair<std::string, int>> sortedWords(wordFreqs.begin(), wordFreqs.end());
    
    // Sort by frequency (descending)
    std::sort(sortedWords.begin(), sortedWords.end(), 
             [](const auto& a, const auto& b) {
                 return a.second > b.second;
             });
    
    // Limit to maxWords
    if (sortedWords.size() > maxWords) {
        sortedWords.resize(maxWords);
    }
    
    return sortedWords;
}

/**
 * @brief Formats a word for display in the word cloud.
 * 
 * This method applies styling to words based on their frequency,
 * such as varying the font weight or using different characters.
 * 
 * @param word The word to format.
 * @param freq The word's frequency.
 * @param maxFreq The maximum frequency in the dataset.
 * @param isPositive Whether to use positive sentiment styling.
 * @return Formatted word string with styling.
 */
std::string AsciiWordCloud::formatWord(
    const std::string& word, 
    int freq, 
    int maxFreq,
    bool isPositive) {
    
    std::ostringstream formatted;
    
    // Apply color coding based on frequency
    formatted << getColorCode(freq, maxFreq, isPositive);
    
    // Format the word based on relative frequency
    double relativeFreq = static_cast<double>(freq) / maxFreq;
    
    if (relativeFreq > 0.8) {
        // Very common words - uppercase and bold
        std::string upperWord = word;
        std::transform(upperWord.begin(), upperWord.end(), upperWord.begin(),
                      [](unsigned char c) { return std::toupper(c); });
        formatted << upperWord;
    } else if (relativeFreq > 0.5) {
        // Common words - uppercase
        std::string upperWord = word;
        std::transform(upperWord.begin(), upperWord.end(), upperWord.begin(),
                      [](unsigned char c) { return std::toupper(c); });
        formatted << upperWord;
    } else if (relativeFreq > 0.3) {
        // Moderately common words - normal
        formatted << word;
    } else {
        // Less common words - small
        formatted << word;
    }
    
    // Reset the color
    formatted << resetColor();
    
    return formatted.str();
}

/**
 * @brief Generates an ANSI color code based on word frequency.
 * 
 * This method selects colors that vary based on the word's frequency
 * and the sentiment context (positive or negative).
 * 
 * @param freq The word's frequency.
 * @param maxFreq The maximum frequency in the dataset.
 * @param isPositive Whether to use positive sentiment coloring.
 * @return ANSI color code string.
 */
std::string AsciiWordCloud::getColorCode(int freq, int maxFreq, bool isPositive) {
    double relativeFreq = static_cast<double>(freq) / maxFreq;
    
    // Scale: 0-6 represents different intensities
    int colorScale = static_cast<int>(relativeFreq * 6);
    
    // ANSI color codes
    if (isPositive) {
        // Blue to green scale for positive sentiment
        switch (colorScale) {
            case 0: return "\033[38;5;39m";  // Light blue
            case 1: return "\033[38;5;38m";
            case 2: return "\033[38;5;37m";
            case 3: return "\033[38;5;36m";
            case 4: return "\033[38;5;35m";
            case 5: return "\033[38;5;34m";
            case 6: return "\033[38;5;46m";  // Bright green
            default: return "\033[38;5;39m"; // Default light blue
        }
    } else {
        // Red to yellow scale for negative sentiment
        switch (colorScale) {
            case 0: return "\033[38;5;196m"; // Bright red
            case 1: return "\033[38;5;202m";
            case 2: return "\033[38;5;208m";
            case 3: return "\033[38;5;214m";
            case 4: return "\033[38;5;220m";
            case 5: return "\033[38;5;226m";
            case 6: return "\033[38;5;227m"; // Yellow
            default: return "\033[38;5;196m"; // Default bright red
        }
    }
}

/**
 * @brief Provides the ANSI code to reset text formatting.
 * @return ANSI reset code string.
 */
std::string AsciiWordCloud::resetColor() {
    return "\033[0m";
}

} // namespace nlp

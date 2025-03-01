/**
 * @file ascii_word_cloud.h
 * @brief Generates ASCII art word clouds from text data.
 * 
 * This module provides functionality to visualize word frequencies as
 * ASCII art, with options for color, size, and different visualization
 * styles. It's useful for exploring the most frequent terms in text
 * collections.
 */

#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <utility>

namespace nlp {

/**
 * @class AsciiWordCloud
 * @brief Generates ASCII-based word clouds from text data.
 * 
 * This class analyzes text, counts word frequencies, and generates
 * visual word clouds that can be displayed in the console or saved
 * to a file.
 */
class AsciiWordCloud {
public:
    /**
     * @struct CloudConfig
     * @brief Configuration options for word cloud generation.
     */
    struct CloudConfig {
        size_t maxWords = 30;      ///< Maximum number of words to include
        size_t width = 80;         ///< Width of the word cloud in characters
        size_t height = 15;        ///< Height of the word cloud in characters
        bool useColor = true;      ///< Whether to use ANSI color codes
        bool useBars = false;      ///< Whether to use bar-style visualization
        bool showFrequencies = false; ///< Whether to show word frequencies
    };

    /**
     * @brief Generates an ASCII word cloud from the given text collection.
     * 
     * This function generates a simple ASCII word cloud with default settings.
     * 
     * @param texts A vector of text inputs to be analyzed.
     * @param maxWords The maximum number of words to include in the word cloud.
     * @param width The width of the ASCII word cloud.
     * @param height The height of the ASCII word cloud.
     * @param isPositive Determines whether this is a positive sentiment cloud (affects styling).
     * @return A formatted string representing the ASCII word cloud.
     */
    static std::string generateWordCloud(
        const std::vector<std::string>& texts,
        size_t maxWords = 30, 
        size_t width = 80, 
        size_t height = 15,
        bool isPositive = true);

    /**
     * @brief Generates a customized ASCII word cloud.
     * 
     * This function provides more control over word cloud generation
     * with a configuration struct for advanced settings.
     * 
     * @param texts A vector of text inputs to be analyzed.
     * @param config Configuration options for the word cloud.
     * @param isPositive Determines sentiment-based styling.
     * @return A formatted string representing the ASCII word cloud.
     */
    static std::string generateCustomCloud(
        const std::vector<std::string>& texts,
        const CloudConfig& config,
        bool isPositive = true);

    /**
     * @brief Displays an ASCII word cloud in the console with color formatting.
     * 
     * This function renders a word cloud directly in the console,
     * applying ANSI colors based on sentiment.
     * 
     * @param texts A vector of text inputs to be analyzed.
     * @param maxWords The maximum number of words to include in the word cloud.
     * @param width The width of the ASCII word cloud.
     * @param height The height of the ASCII word cloud.
     * @param isPositive Determines whether this is a positive sentiment cloud (affects styling).
     */
    static void displayWordCloud(
        const std::vector<std::string>& texts,
        size_t maxWords = 30, 
        size_t width = 80, 
        size_t height = 15,
        bool isPositive = true);

private:
    /**
     * @brief Counts word frequencies in a given collection of texts.
     * 
     * This function tokenizes the provided texts and calculates the 
     * frequency of each word, returning a map of word occurrences.
     * 
     * @param texts A vector of text inputs to be analyzed.
     * @return A map of words and their respective frequencies.
     */
    static std::unordered_map<std::string, int> countWordFrequencies(
        const std::vector<std::string>& texts);

    /**
     * @brief Extracts the top words based on frequency.
     * 
     * This function selects the most frequent words up to a specified limit.
     * 
     * @param wordFreqs A map of word frequencies.
     * @param maxWords The maximum number of top words to extract.
     * @return A sorted vector of (word, frequency) pairs.
     */
    static std::vector<std::pair<std::string, int>> getTopWords(
        const std::unordered_map<std::string, int>& wordFreqs, 
        size_t maxWords);

    /**
     * @brief Formats a word for display in an ASCII word cloud.
     * 
     * This function adjusts word size and applies formatting based on 
     * its frequency.
     * 
     * @param word The word to format.
     * @param freq The frequency of the word.
     * @param maxFreq The maximum frequency in the set.
     * @param isPositive Whether this is for a positive sentiment cloud.
     * @return A formatted word string.
     */
    static std::string formatWord(
        const std::string& word, 
        int freq, 
        int maxFreq,
        bool isPositive);

    /**
     * @brief Retrieves the ANSI color code based on word frequency and sentiment.
     * 
     * @param freq Word frequency.
     * @param maxFreq Maximum frequency in the set.
     * @param isPositive Whether this is for a positive sentiment cloud.
     * @return A string containing the ANSI color code.
     */
    static std::string getColorCode(int freq, int maxFreq, bool isPositive);

    /**
     * @brief Resets the ANSI color formatting.
     * 
     * @return A string containing the ANSI reset code.
     */
    static std::string resetColor();
};

} // namespace nlp

#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <utility>

namespace nlp {

/**
 * @brief Class for generating ASCII-based word clouds
 */
class AsciiWordCloud {
public:
    /**
     * @brief Generate a word cloud from text with a specific sentiment
     * 
     * @param texts Collection of texts with the specified sentiment
     * @param maxWords Maximum number of words to include in the cloud
     * @param width Width of the ASCII display
     * @param height Height of the ASCII display
     * @param isPositive Whether this is a positive sentiment cloud (affects colors)
     * @return std::string Formatted ASCII word cloud
     */
    static std::string generateWordCloud(
        const std::vector<std::string>& texts, 
        size_t maxWords = 30, 
        size_t width = 80, 
        size_t height = 15,
        bool isPositive = true);
    
    /**
     * @brief Output a word cloud to stdout with colors
     * 
     * @param texts Collection of texts with the specified sentiment
     * @param maxWords Maximum number of words to include in the cloud
     * @param width Width of the ASCII display
     * @param height Height of the ASCII display
     * @param isPositive Whether this is a positive sentiment cloud (affects colors)
     */
    static void displayWordCloud(
        const std::vector<std::string>& texts, 
        size_t maxWords = 30, 
        size_t width = 80, 
        size_t height = 15,
        bool isPositive = true);

private:
    /**
     * @brief Count word frequencies in a collection of texts
     * 
     * @param texts Collection of texts
     * @return std::unordered_map<std::string, int> Word frequency map
     */
    static std::unordered_map<std::string, int> countWordFrequencies(
        const std::vector<std::string>& texts);
    
    /**
     * @brief Get a list of top words by frequency
     * 
     * @param wordFreqs Word frequency map
     * @param maxWords Maximum number of words to include
     * @return std::vector<std::pair<std::string, int>> List of (word, frequency) pairs
     */
    static std::vector<std::pair<std::string, int>> getTopWords(
        const std::unordered_map<std::string, int>& wordFreqs, 
        size_t maxWords);
    
    /**
     * @brief Format a word with ASCII art based on its frequency
     * 
     * @param word The word to format
     * @param freq The word's frequency
     * @param maxFreq The maximum frequency in the set
     * @param isPositive Whether this is for a positive sentiment cloud
     * @return std::string Formatted word with ASCII art
     */
    static std::string formatWord(
        const std::string& word, 
        int freq, 
        int maxFreq,
        bool isPositive);

    /**
     * @brief Get ANSI color code for word based on frequency and sentiment
     * 
     * @param freq Word frequency
     * @param maxFreq Maximum frequency in set
     * @param isPositive Whether this is for a positive sentiment cloud
     * @return std::string ANSI color code
     */
    static std::string getColorCode(int freq, int maxFreq, bool isPositive);
    
    /**
     * @brief Reset ANSI color formatting
     * 
     * @return std::string ANSI reset code
     */
    static std::string resetColor();
};

} // namespace nlp
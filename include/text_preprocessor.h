/**
 * @file text_preprocessor.h
 * @brief Handles text preprocessing for sentiment analysis.
 *
 * Provides functions to clean and normalize text by:
 * - Converting to lowercase
 * - Removing punctuation, numbers, and non-ASCII characters
 * - Removing stopwords
 * - Stripping whitespace
 * - Basic word stemming
 * 
 * These preprocessing steps help standardize text for machine learning algorithms,
 * reducing noise and improving feature quality.
 */

#pragma once

#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>
#include <unordered_map>
#include <functional>

namespace nlp {

/**
 * @class TextPreprocessor
 * @brief Prepares text for sentiment analysis.
 *
 * Applies a pipeline of transformations to clean raw text before feature extraction.
 * Each transformation is implemented using a functional approach, making the pipeline
 * composable and testable.
 */
class TextPreprocessor {
public:
    /**
     * @brief Initializes the TextPreprocessor with default stopwords.
     */
    TextPreprocessor();

    /**
     * @brief Cleans and normalizes a given text string.
     * 
     * This method applies a series of text transformations to prepare
     * the input for machine learning algorithms. The optional steps parameter
     * allows customizing the preprocessing pipeline.
     * 
     * @param text The input text to process.
     * @param steps The set of transformations to apply. If empty, default steps are used.
     * @return The preprocessed text.
     */
    std::string preprocess(std::string_view text, 
                          const std::vector<std::string>& steps = {}) const;

    /**
     * @brief Gets the mapping of preprocessing function names to implementations.
     * 
     * This method provides access to individual preprocessing functions,
     * allowing for testing and custom pipeline construction.
     * 
     * @return A map of function names to function implementations.
     */
    std::unordered_map<std::string, std::function<std::string(std::string_view)>> 
    getPreprocessingFunctions() const;

private:
    /**
     * @brief Type alias for text transformation functions.
     */
    using PreprocessingFunc = std::function<std::string(std::string_view)>;

    /**
     * @brief Converts text to lowercase.
     * @param text Input text.
     * @return Lowercase text.
     */
    std::string lowercase(std::string_view text) const;

    /**
     * @brief Removes punctuation characters.
     * @param text Input text.
     * @return Text without punctuation.
     */
    std::string removePunctuation(std::string_view text) const;

    /**
     * @brief Removes numerical digits.
     * @param text Input text.
     * @return Text without numbers.
     */
    std::string removeNumbers(std::string_view text) const;

    /**
     * @brief Removes non-ASCII characters.
     * @param text Input text.
     * @return Text with only ASCII characters.
     */
    std::string removeNonAscii(std::string_view text) const;

    /**
     * @brief Normalizes whitespace.
     * @param text Input text.
     * @return Text with normalized whitespace.
     */
    std::string stripWhitespace(std::string_view text) const;

    /**
     * @brief Removes common stopwords.
     * @param text Input text.
     * @return Text without stopwords.
     */
    std::string removeStopwords(std::string_view text) const;

    /**
     * @brief Applies basic stemming to words.
     * @param text Input text.
     * @return Text with stemmed words.
     */
    std::string stemWords(std::string_view text) const;

    /**
     * @brief Set of stopwords to remove during preprocessing.
     */
    std::unordered_set<std::string> stopwords;
};

}  // namespace nlp

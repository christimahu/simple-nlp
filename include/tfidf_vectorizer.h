/**
 * @file tfidf_vectorizer.h
 * @brief Defines the TF-IDF vectorization class for text feature extraction.
 *
 * This class provides methods to convert raw text into numerical feature
 * vectors using Term Frequency-Inverse Document Frequency (TF-IDF).
 * TF-IDF captures the importance of words in documents relative to a corpus.
 */

#pragma once

#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>
#include <unordered_set>

namespace nlp {

/**
 * @class TfidfVectorizer
 * @brief A class to convert text data into TF-IDF feature vectors.
 *
 * This class tokenizes input text, builds a vocabulary, and computes
 * TF-IDF scores for given documents. TF-IDF helps measure the importance
 * of words relative to a collection of documents by scaling term frequency
 * by the inverse document frequency.
 */
class TfidfVectorizer {
public:
    /**
     * @brief Constructs a TF-IDF vectorizer with customizable parameters.
     * @param sublinearTf If true, applies sublinear term frequency scaling (log).
     * @param maxDf Maximum document frequency threshold for vocabulary pruning.
     * @param maxFeatures Maximum number of features to retain in the vocabulary.
     */
    TfidfVectorizer(bool sublinearTf = true, double maxDf = 0.5, size_t maxFeatures = 5000);

    /**
     * @brief Tokenizes a text string into words.
     * @param text The input text.
     * @return A vector of tokenized words.
     */
    std::vector<std::string> tokenize(std::string_view text) const;

    /**
     * @brief Fits the vectorizer to a corpus of documents.
     * 
     * This method builds the vocabulary and calculates document frequencies needed for TF-IDF.
     * 
     * @param texts A collection of documents to analyze.
     * @param maxDf Maximum document frequency for feature filtering.
     * @param maxFeatures Maximum vocabulary size.
     */
    void fit(const std::vector<std::string>& texts, 
             double maxDf = 0.5, 
             size_t maxFeatures = 5000);

    /**
     * @brief Transforms documents into TF-IDF feature vectors.
     * 
     * This method converts a collection of documents into a matrix of TF-IDF features,
     * using the vocabulary and document frequencies from a previous fit() call.
     * 
     * @param texts A collection of documents to transform.
     * @return A matrix of TF-IDF features (one vector per document).
     */
    std::vector<std::vector<double>> transform(const std::vector<std::string>& texts) const;

    /**
     * @brief Fits the vectorizer to data and transforms it in one step.
     * 
     * This is a convenience method that calls fit() followed by transform()
     * on the same data.
     * 
     * @param texts A collection of documents to fit and transform.
     * @return A matrix of TF-IDF features (one vector per document).
     */
    std::vector<std::vector<double>> fitTransform(const std::vector<std::string>& texts);

private:
    /**
     * @brief Builds vocabulary from tokenized documents.
     * @param tokenizedDocs Collection of tokenized documents.
     * @param maxDf Maximum document frequency threshold.
     * @param maxFeatures Maximum vocabulary size.
     */
    void buildVocabulary(const std::vector<std::vector<std::string>>& tokenizedDocs,
                         double maxDf,
                         size_t maxFeatures);

    /**
     * @brief Calculates TF-IDF score for a term in a document.
     * @param termFreq Frequency of the term in the document.
     * @param docFreq Number of documents containing the term.
     * @param totalDocs Total number of documents in the corpus.
     * @return TF-IDF score.
     */
    double calculateTfIdf(int termFreq, int docFreq, int totalDocs) const;

    bool sublinearTf;                     ///< Whether to apply sublinear TF scaling
    double maxDf;                         ///< Maximum document frequency for feature filtering
    size_t maxFeatures;                   ///< Maximum number of features in vocabulary
    std::unordered_map<std::string, int> vocabulary;  ///< Maps terms to feature indices
    std::vector<int> documentFrequencies; ///< Document frequency for each term
    int totalDocuments;                   ///< Total number of documents seen during fit
};

} // namespace nlp

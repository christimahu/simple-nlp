/**
 * @file tfidf_vectorizer_test.cpp
 * @brief Tests for the TF-IDF vectorizer component
 */

#include "tfidf_vectorizer.h"
#include <iostream>
#include <cassert>
#include <string>
#include <vector>

// Simple test framework
#define TEST(name) void name()
#define RUN_TEST(name) std::cout << "Running " << #name << "... "; name(); std::cout << "PASSED" << std::endl

/**
 * Test TfidfVectorizer functionality.
 * 
 * This function tests the feature extraction component, ensuring
 * that it correctly transforms text into numerical feature vectors.
 * It demonstrates a functional approach to testing vector properties.
 */
TEST(testTfidfVectorizer) {
    nlp::TfidfVectorizer vectorizer;
    
    // Create a small corpus
    std::vector<std::string> corpus = {
        "this is the first document",
        "this document is the second document",
        "and this is the third one",
        "is this the first document"
    };
    
    // Fit and transform
    std::vector<std::vector<double>> features = vectorizer.fitTransform(corpus);
    
    // Check dimensions
    assert(features.size() == corpus.size());
    
    // Check that all feature vectors have the same length using recursion
    const auto checkFeatureSize = [](auto& self, const auto& features, 
                                size_t index, size_t expectedSize) -> bool {
        // Base case: all features checked
        if (index >= features.size()) {
            return true;
        }
        
        // Check this feature vector's size
        if (features[index].size() != expectedSize) {
            return false;
        }
        
        // Recursively check next feature vector
        return self(self, features, index + 1, expectedSize);
    };
    
    size_t featureSize = features[0].size();
    assert(checkFeatureSize(checkFeatureSize, features, 0, featureSize));
    
    // Recursively check for non-zero values (at least one feature should be non-zero)
    const auto hasNonZeroValue = [](auto& self, const auto& feature, size_t index) -> bool {
        // Base case: no non-zero value found
        if (index >= feature.size()) {
            return false;
        }
        
        // Check if this feature is non-zero
        if (std::abs(feature[index]) > 1e-10) {
            return true;
        }
        
        // Recursively check next feature
        return self(self, feature, index + 1);
    };
    
    // Check that each document has at least one non-zero feature
    // Fixed: Properly capturing hasNonZeroValue in the lambda
    const auto checkAllFeaturesHaveValues = [hasNonZeroValue, &features](auto& self, size_t index) -> bool {
        // Base case: all features checked
        if (index >= features.size()) {
            return true;
        }
        
        // Check this feature vector
        if (!hasNonZeroValue(hasNonZeroValue, features[index], 0)) {
            return false;
        }
        
        // Recursively check next feature vector
        return self(self, index + 1);
    };
    
    assert(checkAllFeaturesHaveValues(checkAllFeaturesHaveValues, 0) && "Features should have non-zero values");
    
    // Check that transformation of new documents works
    std::vector<std::string> newDocs = {"this is a new document"};
    std::vector<std::vector<double>> newFeatures = vectorizer.transform(newDocs);
    
    assert(newFeatures.size() == newDocs.size());
    assert(newFeatures[0].size() == featureSize);
}

int main() {
    std::cout << "==== TfidfVectorizer Tests ====" << std::endl;
    testTfidfVectorizer();
    std::cout << "All TfidfVectorizer tests passed!" << std::endl;
    return 0;
}

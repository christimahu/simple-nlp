# C++ Sentiment Analysis Library

This is a C++ implementation of a sentiment analysis system for Twitter data, ported from the Python original. The library provides functionality for text preprocessing, feature extraction, model training, and sentiment prediction.

## Features

- Text preprocessing including stopword removal, punctuation removal, and normalization
- TF-IDF feature extraction
- SGD classifier for sentiment analysis
- Model evaluation tools including confusion matrix and classification reports
- Command-line application for analyzing CSV datasets

## Project Structure

```
nlp_sentiment/
├── CMakeLists.txt           # Build configuration
├── include/
│   └── sentiment_analysis.h # Main header file
├── src/
│   ├── text_preprocessor.cpp
│   ├── tfidf_vectorizer.cpp
│   ├── sgd_classifier.cpp
│   ├── model_evaluator.cpp
│   ├── sentiment_analyzer.cpp
│   └── nlp_main.cpp         # Main application
├── tests/
│   └── nlp_tests.cpp        # Unit tests
└── data/
    └── twitter_data.csv     # Sample data (not included)
```

## Requirements

- C++17 compatible compiler (GCC 8+, Clang 7+, MSVC 2019+)
- CMake 3.10 or higher
- Twitter sentiment dataset (CSV format with sentiment_label,tweet_text columns)

## Building

```bash
# Create a build directory
mkdir build && cd build

# Configure
cmake ..

# Build
cmake --build .

# Run tests
ctest
```

## Usage

### Command-line Interface

```bash
# Run with default dataset path (../data/twitter_data.csv)
./bin/nlp_main

# Run with custom dataset path
./bin/nlp_main /path/to/your/dataset.csv
```

### Library API

```cpp
#include "sentiment_analysis.h"

// Example usage
int main() {
    nlp::SentimentAnalyzer analyzer;
    
    // Load and preprocess data
    analyzer.loadData("data.csv");
    analyzer.preprocessData();
    
    // Extract features and split data
    analyzer.extractFeatures();
    analyzer.splitData();
    
    // Train and evaluate model
    analyzer.trainModel();
    analyzer.evaluateModel();
    
    // Predict sentiment
    auto result = analyzer.predictSentiment("I love this product!");
    std::cout << "Sentiment: " << result["sentiment"] << std::endl;
    
    return 0;
}
```

## Dataset Format

The expected CSV format is:

```
sentiment_label,tweet_text
0,"This product is terrible! Would not recommend."
4,"Absolutely love this, best purchase ever!"
...
```

Where sentiment labels are:
- 0 = Negative sentiment
- 4 = Positive sentiment

## Customization

You can customize the preprocessing steps, feature extraction parameters, and model parameters:

```cpp
// Custom preprocessing steps
analyzer.preprocessData({"lowercase", "remove_punctuation", "remove_stopwords"});

// Custom feature extraction
analyzer.extractFeatures(0.7, 10000);  // maxDf=0.7, maxFeatures=10000

// Custom model parameters
analyzer.trainModel(1e-5, 0.001);  // alpha=1e-5, eta0=0.001
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

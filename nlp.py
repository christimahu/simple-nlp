#!/usr/bin/env python3
"""
Sentiment Analysis Implementation

This script performs sentiment analysis on a dataset of tweets.
It includes preprocessing, visualization with word clouds, and model training.
Designed to be run as a standalone Python script rather than in a notebook.

Original notebook included:
- Data loading and preprocessing
- Word cloud visualization for positive and negative sentiment
- TF-IDF feature extraction
- SGD classifier training and evaluation
"""

import os
import string
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Try to fix SSL certificate issues that might occur with NLTK downloads
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Optional: Use NLTK resources if available, otherwise use simplified approach
try:
    import nltk
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
    print("NLTK resources loaded successfully.")
except Exception as e:
    print(f"NLTK resources could not be loaded: {e}")
    print("Using simplified preprocessing instead.")
    NLTK_AVAILABLE = False


class TextPreprocessor:
    """
    Handles text preprocessing steps for sentiment analysis.
    Includes methods for cleaning and normalizing text data.
    """
    
    def __init__(self):
        """Initialize the preprocessor with necessary resources."""
        self.nltk_available = NLTK_AVAILABLE
        
        # Initialize stopwords either from NLTK or a basic list
        if self.nltk_available:
            self.stops = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
        else:
            # Basic English stopwords if NLTK is not available
            self.stops = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
                "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 
                'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 
                'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 
                'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 
                "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 
                'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 
                'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 
                'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 
                'between', 'into', 'through', 'during', 'before', 'after', 'above', 
                'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 
                'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 
                'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 
                'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 
                'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 
                'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 
                're', 've', 'y'}
    
    def preprocess(self, text, steps=None):
        """
        Apply a series of preprocessing steps to clean and normalize text.
        
        Parameters:
        -----------
        text : str
            The input text to preprocess
        steps : list, optional
            List of preprocessing steps to apply. If None, all steps are applied.
            
        Returns:
        --------
        str
            The preprocessed text
        """
        if steps is None:
            steps = ['remove_non_ascii', 'lowercase', 'remove_punctuation', 
                     'remove_numbers', 'strip_whitespace', 'remove_stopwords']
            
            # Only add stemming if NLTK is available
            if self.nltk_available:
                steps.append('stem_words')
        
        for step in steps:
            if step == 'remove_non_ascii':
                # Remove non-ASCII characters
                text = ''.join([x for x in text if ord(x) < 128])
            
            elif step == 'lowercase':
                # Convert text to lowercase
                text = text.lower()
            
            elif step == 'remove_punctuation':
                # Remove punctuation
                punct_exclude = set(string.punctuation)
                text = ''.join(char for char in text if char not in punct_exclude)
            
            elif step == 'remove_numbers':
                # Remove numerical digits
                text = ''.join([char for char in text if not char.isdigit()])
            
            elif step == 'remove_stopwords':
                # Remove common stopwords
                word_list = text.split(' ')
                text_words = [word for word in word_list if word not in self.stops]
                text = ' '.join(text_words)
            
            elif step == 'stem_words' and self.nltk_available:
                # Lemmatize words to their base form
                word_list = text.split(' ')
                stemmed_words = [self.lemmatizer.lemmatize(word) for word in word_list]
                text = ' '.join(stemmed_words)
            
            elif step == 'strip_whitespace':
                # Remove extra whitespace
                text = ' '.join(text.split())
                
        return text


class SentimentAnalyzer:
    """
    Main class for performing sentiment analysis.
    Handles data loading, preprocessing, visualization, and model training.
    """
    
    def __init__(self, data_path=None):
        """
        Initialize the sentiment analyzer.
        
        Parameters:
        -----------
        data_path : str, optional
            Path to the tweet dataset CSV file
        """
        self.preprocessor = TextPreprocessor()
        self.data_path = data_path
        self.df = None
        self.vectorizer = None
        self.model = None
        
    def load_data(self, data_path=None, from_url=False):
        """
        Load the tweet dataset from a file or URL.
        
        Parameters:
        -----------
        data_path : str, optional
            Path to the CSV file or URL
        from_url : bool, default=False
            Whether to load from a URL or local file
            
        Returns:
        --------
        pandas.DataFrame
            The loaded dataset
        """
        if data_path is not None:
            self.data_path = data_path
        
        if self.data_path is None:
            raise ValueError("No data path provided")
        
        try:
            if from_url:
                # Try to load from URL
                self.df = pd.read_csv(self.data_path, sep=",")
            else:
                # Try to load from local file
                self.df = pd.read_csv(self.data_path, sep=",")
            
            # Set column names if needed
            if self.df.shape[1] == 2:
                self.df.columns = ["sentiment_label", "tweet_text"]
            
            print(f"Loaded dataset with {self.df.shape[0]} rows and {self.df.shape[1]} columns")
            print(self.df.head())
            
            return self.df
        
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def preprocess_data(self, steps=None):
        """
        Preprocess the tweets in the dataset.
        
        Parameters:
        -----------
        steps : list, optional
            List of preprocessing steps to apply
            
        Returns:
        --------
        pandas.DataFrame
            The preprocessed dataset
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Apply preprocessing to each tweet
        self.df['clean_tweet'] = self.df['tweet_text'].apply(
            lambda s: self.preprocessor.preprocess(s, steps)
        )
        
        print("Preprocessing complete. Sample of preprocessed tweets:")
        print(self.df[['tweet_text', 'clean_tweet']].head())
        
        return self.df
    
    def generate_wordcloud(self, sentiment_value, output_path=None):
        """
        Generate a word cloud for tweets with the specified sentiment.
        
        Parameters:
        -----------
        sentiment_value : int
            The sentiment label value (0 for negative, 4 for positive)
        output_path : str, optional
            Path to save the word cloud image
            
        Returns:
        --------
        matplotlib.figure.Figure
            The word cloud figure
        """
        if self.df is None or 'clean_tweet' not in self.df.columns:
            raise ValueError("No preprocessed data available")
        
        # Join all tweets with the specified sentiment
        sentiment_name = "positive" if sentiment_value == 4 else "negative"
        clean_string = ','.join(self.df.loc[self.df['sentiment_label'] == sentiment_value, 'clean_tweet'])
        
        # Create word cloud
        wordcloud = WordCloud(
            max_words=50,
            width=2500,
            height=1500,
            background_color='black',
            stopwords=STOPWORDS
        ).generate(clean_string)
        
        # Plot word cloud
        fig = plt.figure(figsize=(20, 10), facecolor='k', edgecolor='k')
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)
        
        # Save if output path is provided
        if output_path:
            plt.savefig(output_path)
            print(f"Word cloud saved to {output_path}")
        
        plt.title(f"{sentiment_name.capitalize()} Sentiment Word Cloud", color='white')
        return fig
    
    def extract_features(self, max_df=0.5, max_features=6228):
        """
        Extract TF-IDF features from preprocessed tweets.
        
        Parameters:
        -----------
        max_df : float, default=0.5
            Ignore terms that appear in more than this fraction of documents
        max_features : int, default=6228
            Only use top max_features terms ordered by term frequency
            
        Returns:
        --------
        scipy.sparse.csr_matrix
            The TF-IDF feature matrix
        """
        if self.df is None or 'clean_tweet' not in self.df.columns:
            raise ValueError("No preprocessed data available")
        
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            max_df=max_df,
            max_features=max_features,
            stop_words='english'
        )
        
        # Fit and transform the preprocessed tweets
        clean_texts = self.df['clean_tweet']
        tf_idf_features = self.vectorizer.fit_transform(clean_texts)
        
        print(f"Extracted {tf_idf_features.shape[1]} features from {tf_idf_features.shape[0]} tweets")
        
        return tf_idf_features
    
    def split_data(self, features, test_size=40000, random_state=42):
        """
        Split the dataset into training and testing sets.
        
        Parameters:
        -----------
        features : scipy.sparse.csr_matrix
            The feature matrix
        test_size : int or float, default=40000
            Size of the test set
        random_state : int, default=42
            Random seed for reproducibility
            
        Returns:
        --------
        tuple
            (X_train, X_test, y_train, y_test)
        """
        if self.df is None:
            raise ValueError("No data loaded")
        
        # Get the target labels
        y_targets = np.array(self.df['sentiment_label'])
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            features, y_targets, test_size=test_size, random_state=random_state
        )
        
        print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train, alpha=2.9e-05, eta0=0.00164):
        """
        Train a sentiment classifier using SGD.
        
        Parameters:
        -----------
        X_train : scipy.sparse.csr_matrix
            Training feature matrix
        y_train : numpy.ndarray
            Training labels
        alpha : float, default=2.9e-05
            Regularization parameter
        eta0 : float, default=0.00164
            Initial learning rate
            
        Returns:
        --------
        sklearn.linear_model.SGDClassifier
            The trained model
        """
        # Create and train the classifier
        self.model = SGDClassifier(
            loss='modified_huber',
            learning_rate='adaptive',
            penalty='elasticnet',
            alpha=alpha,
            eta0=eta0
        )
        
        self.model.fit(X_train, y_train)
        
        print("Model training complete")
        
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the trained sentiment classifier.
        
        Parameters:
        -----------
        X_test : scipy.sparse.csr_matrix
            Test feature matrix
        y_test : numpy.ndarray
            Test labels
            
        Returns:
        --------
        dict
            Dictionary containing evaluation metrics
        """
        if self.model is None:
            raise ValueError("No trained model available")
        
        # Get predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate accuracy
        score = self.model.score(X_test, y_test)
        print(f"Model Score: {score:.6f}")
        
        # Generate confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(conf_matrix)
        
        # Generate detailed classification report
        class_report = classification_report(
            y_test, y_pred, target_names=['Negative (0)', 'Positive (4)']
        )
        print("\nClassification Report:")
        print(class_report)
        
        # Return evaluation metrics
        return {
            'accuracy': score,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report
        }
    
    def predict_sentiment(self, text):
        """
        Predict the sentiment of new text.
        
        Parameters:
        -----------
        text : str
            The text to analyze
            
        Returns:
        --------
        dict
            Dictionary containing the prediction and confidence
        """
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model and vectorizer must be trained first")
        
        # Preprocess the text
        clean_text = self.preprocessor.preprocess(text)
        
        # Extract features
        features = self.vectorizer.transform([clean_text])
        
        # Make prediction
        sentiment = self.model.predict(features)[0]
        
        # Get confidence scores
        decision_values = self.model.decision_function(features)[0]
        confidence = abs(decision_values)
        
        # Interpret the prediction
        sentiment_label = "Positive" if sentiment == 4 else "Negative"
        
        return {
            'text': text,
            'clean_text': clean_text,
            'sentiment': sentiment_label,
            'confidence': confidence
        }


def run_full_analysis(data_path):
    """
    Run the complete sentiment analysis pipeline.
    
    Parameters:
    -----------
    data_path : str
        Path to the dataset
        
    Returns:
    --------
    SentimentAnalyzer
        The trained analyzer
    """
    print("====== Starting Sentiment Analysis ======")
    
    # Initialize analyzer and load data
    analyzer = SentimentAnalyzer(data_path)
    analyzer.load_data()
    
    # Preprocess the data
    analyzer.preprocess_data()
    
    # Generate word clouds
    print("\n====== Generating Word Clouds ======")
    analyzer.generate_wordcloud(4, "positive_wordcloud.png")
    analyzer.generate_wordcloud(0, "negative_wordcloud.png")
    
    # Extract features and split data
    print("\n====== Extracting Features ======")
    features = analyzer.extract_features()
    X_train, X_test, y_train, y_test = analyzer.split_data(features)
    
    # Train the model
    print("\n====== Training Model ======")
    analyzer.train_model(X_train, y_train)
    
    # Evaluate the model
    print("\n====== Evaluating Model ======")
    analyzer.evaluate_model(X_test, y_test)
    
    # Example predictions
    print("\n====== Example Predictions ======")
    examples = [
        "I absolutely love this new product! It's amazing!",
        "This is the worst experience I've ever had. Terrible service.",
        "The weather is quite nice today, isn't it?",
        "I'm not sure how I feel about this movie."
    ]
    
    for example in examples:
        result = analyzer.predict_sentiment(example)
        print(f"Text: {result['text']}")
        print(f"Sentiment: {result['sentiment']} (confidence: {result['confidence']:.4f})")
        print()
    
    print("====== Analysis Complete ======")
    return analyzer


def run_tests():
    """
    Run inline tests to verify the functionality of the components.
    """
    print("====== Running Tests ======")
    
    # Test preprocessor
    print("\nTesting TextPreprocessor...")
    preprocessor = TextPreprocessor()
    test_text = "@user I LOVED the movie!!! It's amazing & worth $12.99 :) #mustwatch"
    
    processed = preprocessor.preprocess(test_text)
    print(f"Original: {test_text}")
    print(f"Processed: {processed}")
    
    assert isinstance(processed, str), "Preprocessor should return a string"
    assert processed.islower(), "Text should be lowercase after preprocessing"
    assert "$" not in processed, "Punctuation should be removed"
    assert "12" not in processed, "Numbers should be removed"
    
    # Test basic SentimentAnalyzer initialization
    print("\nTesting SentimentAnalyzer initialization...")
    analyzer = SentimentAnalyzer()
    assert analyzer is not None, "Analyzer should be initialized"
    assert hasattr(analyzer, 'preprocessor'), "Analyzer should have a preprocessor"
    
    print("\nAll tests passed!")
    print("====== Tests Complete ======")


if __name__ == "__main__":
    # Run tests to verify component functionality
    run_tests()
    
    # If a data file exists, run the full analysis
    # Otherwise, just show a message about how to use the script
    default_path = "../data/twitter_data.csv"
    
    if os.path.exists(default_path):
        print(f"\nFound data file at {default_path}")
        analyzer = run_full_analysis(default_path)
    else:
        print(f"\nData file not found at {default_path}")
        print("To run the full analysis, place the twitter_data.csv file in the ../data/ directory")
        print("or run the script with a custom path:")
        print("    python sentiment_analysis.py <path_to_data>")
        
        # Get path from command line arguments if provided
        import sys
        if len(sys.argv) > 1:
            custom_path = sys.argv[1]
            if os.path.exists(custom_path):
                print(f"\nFound data file at {custom_path}")
                analyzer = run_full_analysis(custom_path)
            else:
                print(f"Data file not found at {custom_path}")

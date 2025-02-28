```markdown
# NLP Sentiment Analysis Test Results

## Running NLP Sentiment Analysis Tests
```

```
Running testTextPreprocessor... PASSED
Running testTfidfVectorizer... PASSED
Running testSGDClassifier...
Class distribution in training data:
Class 0 (Negative): 4 samples
Class 4 (Positive): 4 samples
Training with Averaged Perceptron algorithm...
Epoch 1/5, Mistakes: 1, Error rate: 0.125
Epoch 2/5, Mistakes: 0, Error rate: 0
Epoch 3/5, Mistakes: 0, Error rate: 0
Epoch 4/5, Mistakes: 0, Error rate: 0
Epoch 5/5, Mistakes: 0, Error rate: 0
Weight statistics:
Average weight: -0.045
Max absolute weight: 0.05
Nonzero weights: 2 out of 2
Intercept (bias): -0.01
PASSED
Running testModelEvaluator... PASSED
Running testSentimentAnalyzer...
Error: Could not open file nonexistent_file.csv
Error: No data loaded. Call loadData() first.
PASSED
All tests passed!
```

## Running NLP Sentiment Analysis on Dataset

```
Found data file at ../data/twitter_data.csv
====== Starting Sentiment Analysis ======
Loaded dataset with 160000 rows
```

### Sample Data
```
Sentiment: 4, Text: "@elephantbird Hey dear, Happy Friday to You  Already had your rice's bowl for lunch ?"
Sentiment: 4, Text: "Ughhh layin downnnn    Waiting for zeina to cook breakfast"
Sentiment: 0, Text: "@greeniebach I reckon he'll play, even if he's not 100%...but i know nothing!! ;) It won't be the same without him."
Sentiment: 0, Text: "@vaLewee I know!  Saw it on the news!"
Sentiment: 0, Text: "very sad that http://www.fabchannel.com/ has closed down. One of the few web services that I've used for over 5 years"
```

## Word Cloud Generation

### Positive Word Cloud
![Positive Word Cloud](Screenshot 2025-02-27 at 8.19.11 PM.png)

```
Word frequency visualization for 80000 texts with positive sentiment.

Top Words:
good (6201)
love (4709)
day (4557)
like (3740)
get (3720)
thanks (3360)
lol (3280)
going (3095)
time (2960)
today (2950)
```

### Negative Word Cloud
![Negative Word Cloud](Screenshot 2025-02-27 at 8.19.18 PM.png)

```
Word frequency visualization for 80000 texts with negative sentiment.

Top Words:
get (4612)
dont (4454)
cant (4374)
work (4362)
like (4056)
day (3895)
today (3641)
got (3374)
going (3312)
back (3243)
```

## Model Training & Evaluation

```
Extracted 6228 features from 160000 tweets
Overall class distribution:
Class 0: 80000 samples (50%)
Class 4: 80000 samples (50%)
Training set size: 120000, Test set size: 40000
```

### Training Model
```
Training with Averaged Perceptron algorithm...
Epoch 1/5, Mistakes: 38135, Error rate: 0.317792
Epoch 2/5, Mistakes: 36678, Error rate: 0.30565
Epoch 3/5, Mistakes: 36160, Error rate: 0.301333
Epoch 4/5, Mistakes: 36542, Error rate: 0.304517
Epoch 5/5, Mistakes: 36146, Error rate: 0.301217
Model training complete
Training took 23 seconds
```

### Model Evaluation
```
Model Score: 0.753550
Confusion Matrix:
Actual \ Pred  0       4
0              14870   5051
4              4807    15272

Per-class accuracy:
Class 4: 76.06% (15272/20079)
Class 0: 74.64% (14870/19921)

Classification Report:
           precision      recall  f1-score   support
----------------------------------------------------
        Negative (0)        0.76      0.75      0.75     19921
        Positive (4)        0.75      0.76      0.76     20079

         avg / total        0.75      0.75      0.75     40000
```

## Example Predictions

```
Text: I absolutely love this new product! It's amazing!
Cleaned: absolutely love new product amazing
Sentiment: Positive
Raw score: 4.420204
Confidence: 0.988111
```

```
Text: This is the worst experience I've ever had. Terrible service.
Cleaned: worst experience ive ever terrible service
Sentiment: Negative
Raw score: -3.367730
Confidence: 0.033319
```

```
Text: The weather is quite nice today, isn't it?
Cleaned: weather quite nice today isnt
Sentiment: Negative
Raw score: -0.315107
Confidence: 0.421869
```

```
Text: I'm not sure how I feel about this movie.
Cleaned: im sure feel movie
Sentiment: Positive
Raw score: 0.007423
Confidence: 0.501856
```

## Conclusion
```
====== Analysis Complete ======
```



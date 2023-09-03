---

# Bitcoin Price Prediction using Sentiment Analysis

This project aims to predict the price of Bitcoin by leveraging sentiment analysis on data sourced from Twitter, Reddit, and Google News.

## Overview

The model uses sentiment scores and other features from the data sources to predict the price of Bitcoin. The sentiment scores are derived from the flair, TextBlob polarity, TextBlob subjectivity, and the positive and negative scores from the Sentiment Intensity Analyzer (SID).

## Features

- **Data Sources**: Twitter, Reddit, Google News
- **Sentiment Analysis Tools**: TextBlob, Sentiment Intensity Analyzer
- **Model**: LSTM (Long Short-Term Memory)

## Dependencies

- numpy
- pandas
- matplotlib
- sklearn
- keras

## Dataset

The dataset `crypto_data_news_reddit_final.csv` contains the following columns:
- Bitcoin price data (open, high, low, close, volume)
- Litecoin price data (close, volume)
- Ethereum price data (close, volume)
- Sentiment scores from Google News and Reddit

## Model Architecture

- LSTM layer with 64 units
- Dense layer with 1 unit
- RMSprop optimizer with a learning rate of 0.01
- Loss function: Mean Squared Error

## Usage

1. Load the dataset:
```python
org_df = pd.read_csv('crypto_data_news_reddit_final.csv',index_col=0)
```

2. Preprocess the data:
```python
# Combine sentiment scores from Google News and Reddit
# ... [rest of preprocessing code]
```

3. Train the model:
```python
history= model.fit(trainX, trainY, validation_split=0.30, epochs=epochs, batch_size=batch_size, shuffle=False)
```

4. Evaluate the model:
```python
# Calculate RMSE and MAE for train and test sets
# ... [rest of evaluation code]
```

5. Visualize the results:
```python
# Plot the actual vs. predicted Bitcoin prices
# ... [rest of plotting code]
```

## Results

The model's performance can be evaluated using Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE). The results are printed after training.

## Future Work

- Incorporate more data sources for sentiment analysis.
- Experiment with different model architectures and hyperparameters.
- Explore other sentiment analysis tools and techniques.

---

# Sentiment Analysis Project

This project involves sentiment analysis of British Airways reviews. The analysis is divided into two parts: data preparation and sentiment analysis. The project is written in Python 3.11.

## Folder Structure

    part_one/
    ├── log_loss.py
    ├── prepare_data.py
    ├── sentiment_analysis.py
    └── tfidf_count_vec.py

    part_two/

## Part One

### `prepare_data.py`

The `prepare_data.py` script is used to prepare data from British Airways reviews. It includes data cleaning, sentiment analysis using multiple methods, feature extraction, and data transformation. The prepared data is saved for further analysis.

#### Key Steps:
- **Data Cleaning:** Removal of unnecessary text and tokenization.
- **Sentiment Analysis:** Using TextBlob, VADER, and SentiWordNet to calculate sentiment scores.
- **Feature Extraction:** Categorizing reviews, creating interaction features, extracting time-related features, and normalizing star ratings.
- **Model Training:** Training various classification models (Logistic Regression, Random Forest, Gradient Boosting, SVM, Neural Network) using TF-IDF vectorization.

#### Execution:
    python3.11 part_one/prepare_data.py


### `log_loss.py`

The `log_loss.py` script measures the log loss of different classification models on the prepared data.

#### Key Steps:
- **Load Data:** Load the cleaned and prepared data.
- **TF-IDF Vectorization:** Convert text data to TF-IDF features.
- **Model Training:** Train Logistic Regression, Random Forest, Gradient Boosting, and SVM classifiers.
- **Log Loss Calculation:** Calculate and plot log loss for training and testing datasets.

#### Execution:
    python3.11 part_one/log_loss.py


### `tfidf_count_vec.py`

This script is used for TF-IDF and count vectorization of text data.

## Part Two

Currently, this folder is empty. Future expansions or additional analyses can be added here.

## Requirements

- Python 3.11
- pandas
- numpy
- textblob
- nltk
- vaderSentiment
- joblib
- scikit-learn
- gensim
- matplotlib
- statsmodels

## Installation

Install the required packages using pip

    pip install pandas numpy textblob nltk vaderSentiment joblib scikit-learn gensim matplotlib statsmodels


## Dataset

The dataset `British_Airway_Review.csv` should be placed in the root directory before running the scripts.

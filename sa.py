"""
        The script uses the TextBlob library to perform sentiment analysis on the review text. 
        The sentiment polarity is calculated for each review, providing a score that ranges 
        from -1 (very negative) to 1 (very positive). This helps in quantifying the sentiment 
        expressed in the reviews, which is essential for understanding customer satisfaction and feedback trends.

        Reviews and star ratings are classified into sentiment categories: Positive, Negative, and Neutral. 
        This classification aids in quickly summarizing and analyzing the general mood of the feedback.

        Interaction features such as seat_traveller and country_route are created to capture more granular 
        insights from the data. These features combine multiple aspects of the reviews, providing 
        richer information for analysis.

        Time features (day of the week, month, year) are extracted from the review dates to identify any 
        temporal patterns in the reviews. This can be useful for understanding seasonal trends or identifying 
        specific periods with higher or lower satisfaction.

        Star ratings and recommendation status are normalized to standardize the data. A composite sentiment 
        score is then calculated by averaging the normalized stars, recommendation status, and sentiment polarity. 
        This score provides a comprehensive measure of overall satisfaction.

        The script calculates average composite sentiment scores by country and route. This allows for comparative 
        analysis between different regions and routes, helping to identify areas with higher or lower satisfaction.
        
        A weighted composite sentiment score is created to balance the importance of different features. 
        This score is crucial for making informed decisions based on multiple aspects of customer feedback.
        https://www.datacamp.com/tutorial/text-classification-python
"""

import pandas as pd
from textblob import TextBlob
import re
import nltk
from nltk.corpus import stopwords
from nltk.corpus import sentiwordnet as swn
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from joblib import Parallel, delayed
import multiprocessing

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('sentiwordnet')

# Read the CSV file into a DataFrame
df = pd.read_csv('British_Airway_Review.csv')

# Define a function to get the sentiment of a text
def get_sentiment(text):
    """
    Calculate the sentiment polarity of a given text using TextBlob.

    Args:
    text (str): The text to analyze.

    Returns:
    float: Sentiment polarity score ranging from -1 (negative) to 1 (positive).
    """
    blob = TextBlob(text)
    return blob.sentiment.polarity

# Define a function to preprocess text
def preprocess_text(text):
    """
    Preprocess the given text by removing special characters, converting to lowercase,
    tokenizing, removing stopwords, and lemmatizing.

    Args:
    text (str): The text to preprocess.

    Returns:
    list: List of cleaned and lemmatized tokens.
    """
    # Remove special characters
    text = re.sub(r'\W', ' ', text)
    # Replace "Trip Verified |" and "Not Verified |" with empty string 
    text = re.sub(r'Trip Verified ', '', text)
    text = re.sub(r'Not Verified ', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return tokens
# Example function to get sentiment scores using SentiWordNet
def get_sentiment_score(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)

    sentiment_score = 0
    count = 0

    for token in tokens:
        if token.lower() not in stop_words:
            lemma = lemmatizer.lemmatize(token.lower())
            if synsets := list(swn.senti_synsets(lemma)):
                synset = synsets[0]
                sentiment_score += synset.pos_score() - synset.neg_score()
                count += 1

    return 0 if count == 0 else sentiment_score / count

# Function to get VADER sentiment scores
def get_vader_sentiment_score(text):
    return analyzer.polarity_scores(text)['compound']

# Define a function to categorize text based on keywords
def categorize_review(tokens):
    """
    Categorize the review text based on the presence of specific keywords.

    Args:
    tokens (list): List of tokens from the review text.

    Returns:
    str: Category of the review ('Customer Service', 'Refund', 'Delays', 'Booking Issues', 'Others').
    """
# Define the updated categories
    categories = {
        'CustomerService': ['service', 'staff', 'crew', 'attendant'],
        'Refund': ['refund', 'money back', 'reimbursement'],
        'Delays': ['delay', 'late', 'cancel'],
        'BookingIssues': ['booking', 'reservation', 'ticket'],
        'LoyaltyProgram': ['miles', 'loyalty', 'frequent flyer', 'avios'],
        'SafetyConcerns': ['safety', 'security', 'turbulence'],
        'InFlightExperience': ['food', 'meal', 'entertainment', 'seat', 'comfort'],
        'BaggageIssues': ['baggage', 'luggage', 'lost', 'damaged'],
        'GroundServices': ['check-in', 'lounge', 'boarding', 'airport'],
        'TechnicalIssues': ['website', 'app', 'online', 'technical']
    }

    return next(
        (
            category
            for category, keywords in categories.items()
            if any(keyword in tokens for keyword in keywords)
        ),
        'Others',
    )

# Apply the sentiment analysis function to the 'reviews' column
df['SentimentScore'] = df['reviews'].apply(get_sentiment)

# Apply the preprocessing and categorization functions to the 'reviews' column
df['reviews'] = df['reviews'].apply(preprocess_text)
df['Category'] = df['reviews'].apply(categorize_review)
reviews = df['reviews'].apply(lambda x: ' '.join(x))
#df['sentiwordnet_score'] = df['reviews'].join(' ').apply(get_sentiment_score)
#convert list of tokens back to string
df['sentiwordnet_score'] = reviews.apply(get_sentiment_score)
df['SentimentScore2'] =reviews.apply(get_sentiment)

analyzer = SentimentIntensityAnalyzer()
#df['vaderSentimentScore'] = analyzer.polarity_scores(df['reviews'])
# Apply VADER sentiment analysis with parallel processing
num_cores = multiprocessing.cpu_count()
df['vader_sentiment_score'] = Parallel(n_jobs=num_cores)(delayed(get_vader_sentiment_score)(text) for text in reviews)


# Create interaction features
df['seat_traveller'] = df['seat_type'] + '_' + df['type_of_traveller']
df['country_route'] = df['country'] + '_' + df['route']
df['country_category'] =  df['country'] + '_' + df['route'] + '_' + df['Category']

# Extract time-related features from the 'date' column
df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
df['month'] = pd.to_datetime(df['date']).dt.month
df['year'] = pd.to_datetime(df['date']).dt.year

# Normalize star ratings and recommended status
df['normalized_stars'] = df['stars'] / df['stars'].max()
df['normalized_recommended'] = df['recommended'].apply(lambda x: 1 if x == 'yes' else 0)

# Calculate a composite sentiment score
df['composite_sentiment'] = (df['normalized_stars'] + df['normalized_recommended'] + df['SentimentScore']) / 3

# Calculate average composite sentiment by country and route
avg_rating_country = df.groupby('country')['composite_sentiment'].mean().to_dict()
avg_rating_route = df.groupby('route')['composite_sentiment'].mean().to_dict()
df['avg_rating_country'] = df['country'].map(avg_rating_country)
df['avg_rating_route'] = df['route'].map(avg_rating_route)

# Combine normalized features into a final composite score using defined weights
weights = {'normalized_stars': 0.4, 'normalized_recommended': 0.3, 'SentimentScore': 0.3}
df['composite_sentiment'] = (weights['normalized_stars'] * df['normalized_stars'] +
                             weights['normalized_recommended'] * df['normalized_recommended'] +
                             weights['SentimentScore'] * df['SentimentScore'])

# Save the new dataset to a CSV file
df.to_csv('BA_AirlineReviews_with_Sentiment.csv', index=False)

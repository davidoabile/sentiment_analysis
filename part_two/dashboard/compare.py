import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import nltk

# Ensure required nltk resources are downloaded
nltk.download('vader_lexicon')
nltk.download('punkt')

# Load the data
file_path = 'British_Airways_Reviews_Latest.csv'
reviews_df = pd.read_csv(file_path)

# Initialize VADER sentiment intensity analyzer
vader_analyzer = SentimentIntensityAnalyzer()

# Define a function to get VADER compound score
def vader_compound(text):
    return vader_analyzer.polarity_scores(text)['compound']

# Define a function to get TextBlob polarity score
def textblob_polarity(text):
    return TextBlob(text).sentiment.polarity

# Apply VADER and TextBlob sentiment analysis
reviews_df['Vader_Compound'] = reviews_df['reviews'].apply(vader_compound)
reviews_df['TextBlob_Polarity'] = reviews_df['reviews'].apply(textblob_polarity)

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(reviews_df['TextBlob_Polarity'], reviews_df['Vader_Compound'], alpha=0.5, s=1)
plt.xlabel('TextBlob Polarity')
plt.ylabel('Vader Compound')
plt.title('Vader Compound vs TextBlob Polarity')
plt.grid(True)
plt.show()

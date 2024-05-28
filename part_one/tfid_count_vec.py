import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

df = pd.read_csv('British_Airway_Review.csv')

# Preprocess text data
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'Trip Verified ', '', text)
    text = re.sub(r'Not Verified ', '', text)
    text = re.sub(r'Verified Review', '', text)
    text = text.lower()
    tokens = word_tokenize(text)
    #stop_words = set(stopwords.words('english')).union({'long', 'like', 'told', 'flight', 'flights', 'airline', 'airways', 'airway', 'british', 'ba', 'virgin', 'atlantic', 'virginatlantic','verified review', 'trip verified', 'not verified', 'verified'})
    stop_words = set(nltk.corpus.stopwords.words('english')).union({'long', 'like', 'told', 'flight', 'flights', 'airline', 'airways', 'airway', 'british', 'ba', 'virgin', 'atlantic', 'virginatlantic','verified review', 'trip verified', 'not verified', 'verified'})
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)




df['cleaned_reviews'] = df['reviews'].apply(preprocess_text)

# TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(df['cleaned_reviews'])
tfidf_features = tfidf_vectorizer.get_feature_names_out()

# Count Vectorizer
count_vectorizer = CountVectorizer(ngram_range=(1, 3), min_df=2, stop_words='english')
X_counts = count_vectorizer.fit_transform(df['cleaned_reviews'])
count_features = count_vectorizer.get_feature_names_out()

# Plot comparison
plt.figure(figsize=(15, 5))

# Plot TF-IDF features
plt.subplot(1, 2, 1)
tfidf_feature_counts = X_tfidf.sum(axis=0).A1
tfidf_top_indices = tfidf_feature_counts.argsort()[-10:][::-1]
plt.bar(tfidf_features[tfidf_top_indices], tfidf_feature_counts[tfidf_top_indices])
plt.title('Top TF-IDF Features')
plt.xticks(rotation=90)

# Plot CountVectorizer features
plt.subplot(1, 2, 2)
count_feature_counts = X_counts.sum(axis=0).A1
count_top_indices = count_feature_counts.argsort()[-10:][::-1]
plt.bar(count_features[count_top_indices], count_feature_counts[count_top_indices])
plt.title('Top CountVectorizer Features')
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()

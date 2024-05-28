import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk

# Ensure nltk's stopwords are downloaded
nltk.download('stopwords')

# Sample data
reviews = [
    "The flight was on time and the staff was friendly",
    "I had a terrible experience, the flight was delayed",
    "The food was great, but the seats were uncomfortable",
    "Amazing flight with great service and on-time departure",
    "The worst flight I've ever had, will not recommend"
]

# Create a DataFrame
reviews_df = pd.DataFrame(reviews, columns=['review'])

# Initialize the TF-IDF vectorizer with English stopwords
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the reviews to TF-IDF features
tfidf_matrix = tfidf_vectorizer.fit_transform(reviews_df['review'])

# Convert the TF-IDF matrix to a DataFrame for better readability
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Display the TF-IDF DataFrame
print("TF-IDF Matrix:\n", tfidf_df)

# Summarize the TF-IDF scores by summing them across all documents
tfidf_summary = tfidf_df.sum().sort_values(ascending=False)
print("\nTF-IDF Summary:\n", tfidf_summary)

# Select the top N features (e.g., top 10 features)
top_n_features = tfidf_summary.head(10).index
print("\nTop N Features:\n", top_n_features)

# Create a DataFrame with the top N features for use in regression
top_features_df = tfidf_df[top_n_features]
print("\nTop Features DataFrame:\n", top_features_df)

# Manually categorize the top features into business areas
business_areas = {
    'service': [],
    'timeliness': [],
    'comfort': [],
    'food': [],
    'experience': [],
    'other': []
}

# Mapping of top features to business areas
feature_to_area_mapping = {
    'staff': 'service',
    'service': 'service',
    'friendly': 'service',
    'time': 'timeliness',
    'terrible': 'experience',
    'delayed': 'timeliness',
    'departure': 'timeliness',
    'seats': 'comfort',
    'uncomfortable': 'comfort',
    'flight': 'comfort',
    'food': 'food',
    'great': 'food',
    'amazing': 'experience',
    'experience': 'experience',
    'recommend': 'experience',
    'worst': 'experience',
    'other': 'other'
}

# Assign features to business areas
for feature in top_n_features:
    area = feature_to_area_mapping.get(feature, 'other')  # Default to 'experience' if not found
    business_areas[area].append(feature)
del business_areas['other']
# Display the categorized features
print("\nCategorized Features by Business Area:")
for area, features in business_areas.items():
    print(f"{area.capitalize()}: {', '.join(features)}")

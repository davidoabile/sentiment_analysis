import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk


# Ensure nltk's VADER lexicon is downloaded
nltk.download('vader_lexicon')

# Load the data
file_path = 'British_Airways_Reviews_Latest.csv'
reviews_df = pd.read_csv(file_path)

# Initialize VADER sentiment intensity analyzer
sid = SentimentIntensityAnalyzer()

# Define a function to categorize sentiment
def categorize_sentiment(text):
    #text = ' '.join(text)
    scores = sid.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        return 2  # Positive
    elif compound <= -0.05:
        return 0  # Negative
    else:
        return 1  # Neutral

# Apply sentiment categorization to the reviews
reviews = reviews_df['reviews2'].apply(lambda x: ' '.join(x))
reviews_df['sentiment'] = reviews_df['reviews'].apply(categorize_sentiment)

# Extract the relevant columns
X = reviews_df['reviews']
y = reviews_df['sentiment']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Use TF-IDF for feature extraction
tfidf_vectorizer = TfidfVectorizer(max_features=5000,stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
# Fit and transform the reviews to TF-IDF features
tfidf_matrix = tfidf_vectorizer.fit_transform(reviews_df['reviews'])

# Convert the TF-IDF matrix to a DataFrame for better readability
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
# Summarize the TF-IDF scores by summing them across all documents
tfidf_summary = tfidf_df.sum().sort_values(ascending=False)
# Select the top N features (e.g., top 10 features)
top_n_features = tfidf_summary.head(30).index
print("\nTop N Features:\n", top_n_features)



# Define the classifiers
log_reg = LogisticRegression(max_iter=1000)
random_forest = RandomForestClassifier(n_estimators=100)
gradient_boosting = GradientBoostingClassifier(n_estimators=100)
svm = SVC(kernel='linear', probability=True)
neural_net = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)

# Train the classifiers
log_reg.fit(X_train_tfidf, y_train)
random_forest.fit(X_train_tfidf, y_train)
gradient_boosting.fit(X_train_tfidf, y_train)
svm.fit(X_train_tfidf, y_train)
neural_net.fit(X_train_tfidf, y_train)

# Make predictions
log_reg_preds = log_reg.predict(X_test_tfidf)
random_forest_preds = random_forest.predict(X_test_tfidf)
gradient_boosting_preds = gradient_boosting.predict(X_test_tfidf)
svm_preds = svm.predict(X_test_tfidf)
neural_net_preds = neural_net.predict(X_test_tfidf)

# Combine the predictions
combined_preds = pd.DataFrame({
    'Logistic Regression': log_reg_preds,
    'Random Forest': random_forest_preds,
    'Gradient Boosting': gradient_boosting_preds,
    'SVM': svm_preds,
    'Neural Network': neural_net_preds
})

# This section of code is designed to combine the predictions from different 
# classifiers into a single DataFrame. This can be useful for various ensemble 
# methods, such as majority voting, where the final prediction is based on the 
# majority vote from multiple classifiers.
# Majority voting involves selecting the most common prediction (mode) 
# for each sample across all classifiers.
# [0]: Selects the first mode in case of ties (pandas returns all modes in case of ties).
# Why combine predictions from multiple classifiers?
#   -- Improved Accuracy: Combining multiple models' predictions often leads to better overall 
#       accuracy than using a single model, as different models may capture different patterns in the data.
#   -- Robustness: Ensemble methods like majority voting can make the final prediction 
#       more robust to errors from individual models.
#   -- Reduction of Overfitting: Aggregating predictions from multiple models can reduce the 
#       risk of overfitting, as it smooths out individual models' predictions.
final_preds = combined_preds.mode(axis=1)[0]
sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
reviews_df.loc[X_test.index, 'sentimentValue'] = final_preds.map(sentiment_map)
# Evaluation
report = classification_report(y_test, final_preds, labels=[0, 1, 2], target_names=['Negative', 'Neutral', 'Positive'], zero_division=0)

# Display the combined predictions and the classification report
print("Value\n", final_preds)
print("\nClassification Report:\n", report)
reviews_df.to_csv('British_Airways_Reviews_Latest.csv', index=False)


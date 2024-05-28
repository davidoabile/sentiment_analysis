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
from sklearn.model_selection import train_test_split
import multiprocessing
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import matplotlib.pyplot as plt

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('sentiwordnet')

# Read the CSV file into a DataFrame
df = pd.read_csv('British_Airway_Review.csv')

# Define a function to get the sentiment of a text using TextBlob
def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

# Define a function to preprocess text
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
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)['compound']

# Define a function to categorize text based on keywords
def categorize_review(tokens):
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
        (category for category, keywords in categories.items() if any(keyword in tokens for keyword in keywords)),
        'Others'
    )
# Adjust the original sentiment score based on ensemble predictions
def adjust_sentiment(original_score, ensemble_prediction):
    if ensemble_prediction == 'Positive':
        return max(original_score, 0.01)
    else:
        return min(original_score, -0.01)

# Define a function to categorize sentiment using SentiWordNet
def sentiwordnet_sentiment(text):
    tokens = nltk.word_tokenize(text)
    sentiment_score = 0
    for token in tokens:
        if synsets := list(swn.senti_synsets(token)):
            synset = synsets[0]
            sentiment_score += synset.pos_score() - synset.neg_score()
    return sentiment_score / 10

# Combine the sentiment scores from VADER, TextBlob, and SentiWordNet
def combined_sentiment(text):
    vader = get_vader_sentiment_score(text)
    textblob =  TextBlob(text).sentiment.polarity
    sentiwordnet =  get_sentiment_score(text)
    sentiments = [vader, textblob, sentiwordnet]
    # Majority voting
    min_value = min(set(sentiments), key=sentiments.count)

    # Some models fail to predict a negative sentiment, so we return the min value
    # Even in democracy, the minority has the right to have their opinion
    # In Australia we choose a prime minister with a single vote then they kick him/her out 
    # a choose another with a single vote so blame our country for my code :)
    if min_value < 0 :
        return min_value
    # If the majority of models predict a positive sentiment, return the max value
    # Sometimes citizens win
    return max(set(sentiments), key=sentiments.count)


    
# Apply preprocessing and sentiment analysis functions
df['cleaned_reviews'] = df['reviews'].apply(preprocess_text)
reviews = df['cleaned_reviews'].apply(lambda x: ' '.join(x))
df['SentimentScore'] = df['reviews'].apply(combined_sentiment)
df['composite_sentiment_corrected'] = df['SentimentScore']
# Categorize reviews
df['Category'] = df['reviews'].apply(categorize_review)

# Create interaction features
df['seat_traveller'] = df['seat_type'] + '_' + df['type_of_traveller']
df['country_route'] = df['country'] + '_' + df['route']
df['country_category'] = df['country'] + '_' + df['route'] + '_' + df['Category']

# Extract time-related features
df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
df['month'] = pd.to_datetime(df['date']).dt.month
df['year'] = pd.to_datetime(df['date']).dt.year

# Normalize star ratings and recommended status
df['normalized_stars'] = df['stars'] / df['stars'].max()
df['normalized_recommended'] = df['recommended'].apply(lambda x: 1 if x == 'yes' else 0)

# Calculate average composite sentiment by country and route
avg_rating_country = df.groupby('country')['SentimentScore'].mean().to_dict()
avg_rating_route = df.groupby('route')['SentimentScore'].mean().to_dict()
df['avg_rating_country'] = df['country'].map(avg_rating_country)
df['avg_rating_route'] = df['route'].map(avg_rating_route)

# Combine normalized features into a final composite score using defined weights
weights = {'normalized_stars': 0.4, 'normalized_recommended': 0.3, 'SentimentScore': 0.3}
df['composite_sentiment'] = (weights['normalized_stars'] * df['normalized_stars'] +
                             weights['normalized_recommended'] * df['normalized_recommended'] +
                             weights['SentimentScore'] * df['SentimentScore'])
X = reviews
# Preprocessing pipeline
numeric_features = ['normalized_stars', 'normalized_recommended', 'avg_rating_country', 'avg_rating_route', 'day_of_week', 'month', 'year']
categorical_features = []

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Define models
log_reg = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', LogisticRegression(multi_class='ovr', max_iter=1000))])

random_forest = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', RandomForestClassifier())])

gradient_boosting = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('classifier', GradientBoostingClassifier())])

svm = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', SVC(kernel='rbf', class_weight='balanced'))])


# You can change this to whatever score value you want to use
y_classification = pd.cut(df['SentimentScore'], bins=[-float('inf'), 0, float('inf')],
                          labels=['Negative', 'Positive']).astype('category')

# Split data for classification
X_train, X_test, y_train, y_test = train_test_split(X, y_classification, test_size=0.30, random_state=42, stratify=y_classification)
n = 100

tfidf_vectorizer = TfidfVectorizer(max_features=5000,stop_words='english', ngram_range=(1, 3), min_df=5)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
# Fit and transform the reviews to TF-IDF features
#tfidf_matrix = tfidf_vectorizer.fit_transform(reviews)

#tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
# Summarize the TF-IDF scores by summing them across all documents
#tfidf_summary = tfidf_df.sum().sort_values(ascending=False)
# Select the top N features (e.g., top 10 features)

#top_n_features = tfidf_summary.head(n).index
#print(f"\nTop {n} Features:\n", top_n_features)

# Set up the Count Vectorizer to capture unigrams, bigrams, and trigrams
# Comment this out if you want to use phrase modeling
#count_vectorizer = CountVectorizer(max_features=5000, stop_words='english', ngram_range=(3, 3), min_df=10)
count_vectorizer = CountVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 3), min_df=10)

# Fit and transform the reviews to count features
X_count = count_vectorizer.fit_transform(reviews)

# Convert the count matrix to a DataFrame for easier inspection
count_df = pd.DataFrame(X_count.toarray(), columns=count_vectorizer.get_feature_names_out())

# Display the top N features
count_summary = count_df.sum().sort_values(ascending=False)
top_n_count_features = count_summary.head(n).index
print(f"\nTop {n} Count Features:\n", top_n_count_features)

# Create pseudo-sentences using the top N features
#pseudo_sentences = [[feature] for feature in top_n_count_features]
sentences = df['cleaned_reviews'].tolist()
# Train Word2Vec model using the pseudo-sentences
word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=4, workers=4)

# Get the most similar words to a specific word
try:
    similar_words = word2vec_model.wv.most_similar('late', topn=10)
    print("Most similar words to 'service':")
    for word, similarity in similar_words:
        print(f"{word}: {similarity:.4f}")
except KeyError:
    print("The word 'service' is not in the vocabulary.")

# Create tagged documents using the cleaned reviews
tagged_data = [TaggedDocument(words=doc, tags=[str(i)]) for i, doc in enumerate(df['cleaned_reviews'])]

# Train the Doc2Vec model
doc2vec_model = Doc2Vec(tagged_data, vector_size=100, window=5, min_count=2, workers=4, epochs=100)

# Infer the vector for a specific document (e.g., the first document)
doc_vector = doc2vec_model.infer_vector(['business', 'class', 'ticket', 'price', 'expensive'])

# Find the most similar documents
similar_docs = doc2vec_model.dv.most_similar([doc_vector], topn=5)

# Print the most similar documents
print("\nMost similar documents to the first document:")

for doc_id, similarity in similar_docs:
    original_doc_id = int(doc_id)
    original_review = df.iloc[original_doc_id]['reviews']
    print(f"Document ID: {doc_id}, Similarity: {similarity:.4f}")
    print(f"Review: {original_review}\n")
    


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

final_preds = combined_preds.mode(axis=1)[0]
report = classification_report(y_test, final_preds, target_names=['Negative', 'Positive'], zero_division=0)

# Display the combined predictions and the classification report
print("\nClassification Report:\n", report)


# Adjust decision threshold to improve precision
y_scores = random_forest.predict_proba(X_test_tfidf)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores, pos_label='Positive')

# Plot precision against threshold
plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
plt.xlabel("Threshold")
plt.legend(loc="upper left")
plt.ylim([0, 1])
plt.show()

# Set a higher threshold
#threshold = 0.6
#final_preds = (y_scores > threshold).astype(int)

# Evaluation
#report = classification_report(y_test, final_preds, target_names=['Negative', 'Positive'], zero_division=0)
#print("\nClassification Report:\n", report)


# Align indices of final_preds with X_test
final_preds.index = df.index[df.index.isin(X_test.index)]


try:
    # Apply the adjustment only to the test set
    df.loc[X_test.index, 'composite_sentiment_corrected'] = df.loc[X_test.index].apply(
    lambda row: adjust_sentiment(row['SentimentScore'], final_preds.loc[row.name]), axis=1
)
except KeyError:
    print("Error: The indices in X_test do not match the DataFrame.")
    

df.to_csv('BA_AirlineReviews_with_Sentiment.csv', index=False)

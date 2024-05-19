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
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report

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
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
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

# Apply preprocessing and sentiment analysis functions
df['reviews'] = df['reviews'].apply(preprocess_text)
reviews = df['reviews'].apply(lambda x: ' '.join(x))
df['SentimentScore'] = reviews.apply(get_sentiment)
df['sentiwordnet_score'] = reviews.apply(get_sentiment_score)
df['SentimentScore2'] = reviews.apply(get_sentiment)
analyzer = SentimentIntensityAnalyzer()
num_cores = multiprocessing.cpu_count()
df['vader_sentiment_score'] = Parallel(n_jobs=num_cores)(delayed(get_vader_sentiment_score)(text) for text in reviews)

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

# Calculate composite sentiment score
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

# Prepare data for model training
X = df[['normalized_stars', 'normalized_recommended', 'SentimentScore', 'avg_rating_country', 'avg_rating_route', 'day_of_week', 'month', 'year']]
y = df['composite_sentiment'].apply(lambda x: 'Positive' if x > 0.3 else ('Neutral' if x > 0.1 else 'Negative'))

# Preprocessing pipeline
numeric_features = ['normalized_stars', 'normalized_recommended', 'SentimentScore', 'avg_rating_country', 'avg_rating_route', 'day_of_week', 'month', 'year']
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

# Train and evaluate models
models = {'Logistic Regression': log_reg, 
          'Random Forest': random_forest, 
          'Gradient Boosting': gradient_boosting, 
          'SVM': svm}

best_model = None
best_accuracy = 0
best_model_name = ''
model_stats = {}

for name, model in models.items():
    model.fit(X, y)
    y_pred = model.predict(X)
    report = classification_report(y, y_pred, output_dict=True)
    accuracy = report['accuracy']
    model_stats[name] = report
    # You can print the classification report for each model
    #print(f'{name} Classification Report:')
    #print(classification_report(y, y_pred))
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_name = name
        best_model = model

# Predict sentiment values using the best model
df['sentimentValue'] = best_model.predict(X)

# Save the new dataset to a CSV file
df.to_csv('BA_AirlineReviews_with_Sentiment.csv', index=False)

# Print summary statistics for all models
print("\nBest performing model:", best_model_name)
stats  =  model_stats.get(best_model_name)
print(f"Accuracy: {stats['accuracy']:.2f}")
print(f"Macro Avg Precision: {stats['macro avg']['precision']:.2f}")
print(f"Macro Avg Recall: {stats['macro avg']['recall']:.2f}")
print(f"Macro Avg F1-Score: {stats['macro avg']['f1-score']:.2f}")
print(f"Weighted Avg Precision: {stats['weighted avg']['precision']:.2f}")
print(f"Weighted Avg Recall: {stats['weighted avg']['recall']:.2f}")
print(f"Weighted Avg F1-Score: {stats['weighted avg']['f1-score']:.2f}")

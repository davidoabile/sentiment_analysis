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
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('sentiwordnet')

# Initialize VADER sentiment intensity analyzer
vader_analyzer = SentimentIntensityAnalyzer()

# Define a function to categorize sentiment using VADER
def vader_sentiment(text):
    scores = vader_analyzer.polarity_scores(text)
    return scores['compound']

# Define a function to categorize sentiment using TextBlob
def textblob_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity
    

# Define a function to categorize sentiment using SentiWordNet
def sentiwordnet_sentiment(text):
    tokens = nltk.word_tokenize(text)
    sentiment_score = 0
    for token in tokens:
        if synsets := list(swn.senti_synsets(token)):
            synset = synsets[0]
            sentiment_score += synset.pos_score() - synset.neg_score()
    return sentiment_score / 10

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

# Combine the sentiment scores from VADER, TextBlob, and SentiWordNet
def combined_sentiment(text):
    vader = vader_sentiment(text)
    textblob = textblob_sentiment(text)
    sentiwordnet = sentiwordnet_sentiment(text)
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


def prepare(df, file_name):
    # Apply preprocessing and sentiment analysis functions
    df['reviews2'] = df['reviews'].apply(preprocess_text)
    #reviews = df['reviews2'].apply(lambda x: ' '.join(x))
    # Apply combined sentiment categorization to the reviews
    df['SentimentScore'] = df['reviews'].apply(combined_sentiment)

    #df['SentimentScore'] = reviews.apply(get_sentiment)
    #df['sentiwordnet_score'] = reviews.apply(get_sentiment_score)
    #df['SentimentScore2'] = reviews.apply(get_sentiment)
    #num_cores = multiprocessing.cpu_count()
    #df['vader_sentiment_score'] = Parallel(n_jobs=num_cores)(delayed(get_vader_sentiment_score)(text) for text in reviews)

    # Categorize reviews
    df['Category'] = df['reviews2'].apply(categorize_review)

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
    y = df['SentimentScore'].apply(lambda x: 'Positive' if x > 0.2 else ('Neutral' if x >0  else 'Negative'))

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

    neural_net = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42))])

    # Train and evaluate models
    models = {'Logistic Regression': log_reg, 
            'Random Forest': random_forest, 
            'Gradient Boosting': gradient_boosting, 
            'SVM': svm,
            'Neural Network': neural_net}

    best_model = None
    best_accuracy = 0
    model_stats = {}
    best_model_name = ''

    for name, model in models.items():
        model.fit(X, y)
        y_pred = model.predict(X)
        report = classification_report(y, y_pred, output_dict=True)
        accuracy = report['accuracy']
        model_stats[name] = report
        #print(f'{name} Classification Report:')
        #print(classification_report(y, y_pred))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = name

    # Predict sentiment values using the best model
    df['sentimentValue'] = best_model.predict(X)

    # Save the new dataset to a CSV file
    df.to_csv(file_name, index=False)

    # Print summary statistics for all models
    print("\nSummary of Model Performance:")
    display_model_chosen(best_model_name,  model_stats.get(best_model_name))
    
    return df


def display_model_chosen(name, stats):
    print("\nBest performing model:", name)
    print(f"Accuracy: {stats['accuracy']:.2f}")
    print(f"Macro Avg Precision: {stats['macro avg']['precision']:.2f}")
    print(f"Macro Avg Recall: {stats['macro avg']['recall']:.2f}")
    print(f"Macro Avg F1-Score: {stats['macro avg']['f1-score']:.2f}")
    print(f"Weighted Avg Precision: {stats['weighted avg']['precision']:.2f}")
    print(f"Weighted Avg Recall: {stats['weighted avg']['recall']:.2f}")
    print(f"Weighted Avg F1-Score: {stats['weighted avg']['f1-score']:.2f}")
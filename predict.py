import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Load your data
data = pd.read_csv('BA_AirlineReviews_with_Sentiment_test.csv')

# Define target and features
X = data.drop(['SentimentScore'], axis=1)
y = data['SentimentScore']

# Binning sentiment scores into categories
y_classification = pd.cut(y, bins=[-float('inf'), -0.1, 0.1, float('inf')], labels=['Negative', 'Neutral', 'Positive'])

# Create interaction features
X['seat_traveller'] = X['seat_type'] + '_' + X['type_of_traveller']
X['country_route'] = X['country'] + '_' + X['route']

# Extract time features
X['day_of_week'] = pd.to_datetime(data['date']).dt.dayofweek
X['month'] = pd.to_datetime(data['date']).dt.month
X['year'] = pd.to_datetime(data['date']).dt.year

# Calculate average ratings
avg_rating_country = data.groupby('country')['SentimentScore'].mean().to_dict()
avg_rating_route = data.groupby('route')['SentimentScore'].mean().to_dict()
X['avg_rating_country'] = X['country'].map(avg_rating_country)
X['avg_rating_route'] = X['route'].map(avg_rating_route)

# Define numeric and categorical features
numeric_features = ['avg_rating_country', 'avg_rating_route', 'day_of_week', 'month']
categorical_features = ['country', 'seat_type', 'route', 'type_of_traveller', 'seat_traveller', 'country_route']

# Preprocessing pipeline
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define models with hyperparameters and class weights
log_reg = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', LogisticRegression(multi_class='ovr', max_iter=1000))])

random_forest = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', RandomForestClassifier())])

gradient_boosting = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('classifier', GradientBoostingClassifier())])

svm = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', SVC(kernel='rbf', class_weight='balanced'))])

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y_classification, test_size=0.2, random_state=42, stratify=y_classification)

# Training and evaluation
models = {'Logistic Regression': log_reg, 
          'Random Forest': random_forest, 
          'Gradient Boosting': gradient_boosting, 
          'SVM': svm}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f'{name} Classification Report:')
    print(classification_report(y_test, y_pred, zero_division=1))

# Additional checks for regression assumptions
# For regression, include 'recommended' and 'stars' columns back
X_reg = data[['recommended', 'stars'] + numeric_features + categorical_features]
y_reg = data['SentimentScore']

# One-hot encode categorical features for regression
X_reg = pd.get_dummies(X_reg, columns=categorical_features, drop_first=True)

# Remove any remaining non-numeric columns and handle missing values
X_reg = X_reg.apply(pd.to_numeric, errors='coerce').fillna(0)

# Adding constant for intercept
X_reg = sm.add_constant(X_reg)

# Fitting the OLS regression model
model = sm.OLS(y_reg, X_reg).fit()
print(model.summary())

# Plot histogram
# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(y, bins=30, edgecolor='k', alpha=0.7, label='Sentiment Scores')

# Add vertical lines for bins
plt.axvline(x=-0.1, color='r', linestyle='--', label='Negative/Neutral Boundary (-0.1)')
plt.axvline(x=0.1, color='b', linestyle='--', label='Neutral/Positive Boundary (0.1)')

# Adding the mean lines for each class
negative_mean = y[y_classification == 'Negative'].mean()
neutral_mean = y[y_classification == 'Neutral'].mean()
positive_mean = y[y_classification == 'Positive'].mean()

plt.axvline(x=negative_mean, color='r', linestyle='-', label=f'Mean Negative ({negative_mean:.2f})')
plt.axvline(x=neutral_mean, color='g', linestyle='-', label=f'Mean Neutral ({neutral_mean:.2f})')
plt.axvline(x=positive_mean, color='b', linestyle='-', label=f'Mean Positive ({positive_mean:.2f})')

# Labels and title
plt.xlabel('SentimentScore')
plt.ylabel('Frequency')
plt.title('Distribution of SentimentScore with Bins for Negative, Neutral, and Positive')
plt.legend()
plt.grid(False)
plt.show()
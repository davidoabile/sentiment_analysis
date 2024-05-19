import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Load your data
data = pd.read_csv('BA_AirlineReviews_with_Sentiment_test.csv')

# Define target and features
X = data.drop(['SentimentScore', 'recommended', 'stars'], axis=1)
y = data['SentimentScore']

# Binning sentiment scores into categories
y_classification = pd.cut(y, bins=[-float('inf'), -0.1, 0.1, float('inf')], labels=['Negative', 'Neutral', 'Positive'])

# Define numeric and categorical features
numeric_features = []  # Update with your numeric features if any
categorical_features = [ 'seat_type']

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

# Checking for homoscedasticity
plt.scatter(model.fittedvalues, model.resid)
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted values')
plt.show()

# Breusch-Pagan test for heteroscedasticity
from statsmodels.stats.diagnostic import het_breuschpagan

bp_test = het_breuschpagan(model.resid, X_reg)
print('Breusch-Pagan test:', bp_test)

# Q-Q plot for residuals
sm.qqplot(model.resid, line ='45')
plt.title('Q-Q Plot of Residuals')
plt.show()

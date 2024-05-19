import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan

stats_field = "composite_sentiment"
# Load the data
data = pd.read_csv('BA_AirlineReviews_with_Sentiment.csv')
y = data[stats_field]

# Binning sentiment scores into categories
y_classification = pd.cut(y, bins=[-float('inf'), 0.2, 0.5, float('inf')], labels=['Negative', 'Neutral', 'Positive'], right=True)

# Define target and features for classification
X = data.drop([stats_field, 'normalized_recommended', 'normalized_stars'], axis=1)

# Define numeric and categorical features
numeric_features = ['avg_rating_country', 'avg_rating_route', 'day_of_week', 'month']
categorical_features = ['country', 'seat_type', 'route', 'type_of_traveller', 'seat_traveller', 'country_route', 'country_category']

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
                          ('classifier', LogisticRegression(multi_class='ovr', max_iter=2000))])

random_forest = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', RandomForestClassifier())])

gradient_boosting = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('classifier', GradientBoostingClassifier())])

svm = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', SVC(kernel='rbf', class_weight='balanced'))])

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y_classification, test_size=0.33, random_state=42, stratify=y_classification)

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



# For regression, include 'recommended' and 'stars' columns back
X_reg = data[['normalized_recommended', 'normalized_stars'] + numeric_features + ['seat_traveller', 'country_route', 'country_category']]
y_reg = data[stats_field]

# One-hot encode categorical features for regression
X_reg = pd.get_dummies(X_reg, columns=['seat_traveller', 'country_route', 'country_category'], drop_first=True)

# Remove any remaining non-numeric columns and handle missing values
X_reg = X_reg.apply(pd.to_numeric, errors='coerce').fillna(0)

# Adding constant for intercept
X_reg = sm.add_constant(X_reg)

# Fitting the initial OLS regression model
initial_model = sm.OLS(y_reg, X_reg).fit()

# Removing variables with high p-values (>0.05) from the model, ensure at least two columns remain
high_p_value_columns = initial_model.pvalues[initial_model.pvalues > 0.01].index
X_reg_reduced = X_reg.drop(columns=high_p_value_columns, errors='ignore')

# This code ensures that the constant term is always included in the regression model 
# and handles the case where all variables are removed due to high p-values. 
if 'const' not in X_reg_reduced.columns:
    X_reg_reduced = sm.add_constant(X_reg_reduced)

# Refit the model with reduced variables
final_model = sm.OLS(y_reg, X_reg_reduced).fit()

# Outputting the final model summary
print(final_model.summary())

# Checking for homoscedasticity
plt.figure(figsize=(10, 6))
plt.scatter(final_model.fittedvalues, final_model.resid)
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted values')
#plt.show()

# Performs the Breusch-Pagan test for homoscedasticity.
bp_test = het_breuschpagan(final_model.resid, X_reg_reduced)
print('Breusch-Pagan test:', bp_test)

# Q-Q plot for residuals
sm.qqplot(final_model.resid, line='45')
plt.title('Q-Q Plot of Residuals')
#plt.show()

# Plot histogram to show bins
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
plt.xlabel(stats_field)
plt.ylabel('Frequency')
plt.title('Distribution of Sentiment Score with Bins for Negative, Neutral, and Positive')
plt.legend()
plt.grid(False)
#plt.show()


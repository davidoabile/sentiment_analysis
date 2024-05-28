import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_curve, auc, mean_squared_error, r2_score,log_loss
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import statsmodels.api as sm

# Load data
data = pd.read_csv('BA_AirlineReviews_with_Sentiment.csv')
y = data["composite_sentiment"]

# For classification
y_classification = pd.cut(y, bins=[-float('inf'), 0.22, float('inf')],
                          labels=['Negative', 'Positive']).astype('category')

# Features
X = data.drop(["composite_sentiment", 'normalized_recommended', 'normalized_stars'], axis=1)
numeric_features = ['avg_rating_country', 'avg_rating_route', 'day_of_week', 'month']
categorical_features = ['country_category', 'seat_type', 'route', 'type_of_traveller', 'seat_traveller', 'country_route']

# Preprocessor
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# Pipelines for classification
classifiers = {
    'Logistic Regression': LogisticRegression(multi_class='ovr', max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100),
    'SVM': SVC(kernel='rbf', class_weight='balanced', probability=True)
}

# Pipelines for regression
regressors = {
    'Ridge Regression': Ridge(),
    'Random Forest Reg': RandomForestRegressor(n_estimators=100),
    'Gradient Boosting Reg': GradientBoostingRegressor(n_estimators=100)
}

# Split data for classification
X_train, X_test, y_train_class, y_test_class = train_test_split(X, y_classification, test_size=0.33, random_state=42, stratify=y_classification)
# Split data for regression
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y, test_size=0.33, random_state=42)

# Train and evaluate classifiers
for name, model in classifiers.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])
    pipeline.fit(X_train, y_train_class)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    print(f'{name} Classification Report:\n{classification_report(y_test_class, y_pred)}')

    if y_prob is not None:
        print(f"{name} Loss: {log_loss(y_test_class, y_prob)}")
        fpr, tpr, _ = roc_curve(y_test_class, y_prob[:, 1], pos_label='Positive')
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc(fpr, tpr):.2f})')

# Train and evaluate regressors
for name, model in regressors.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', model)])
    pipeline.fit(X_train_reg, y_train_reg)
    y_pred_reg = pipeline.predict(X_test_reg)
    mse = mean_squared_error(y_test_reg, y_pred_reg)
    r2 = r2_score(y_test_reg, y_pred_reg)
    print(f'{name} Regression MSE: {mse:.2f}, R2: {r2:.2f}')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False positive rate (FPR)')
plt.ylabel('True positive rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='best')
plt.show()

# Train Random Forest and Gradient Boosting models
rf_model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])
gb_model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', GradientBoostingClassifier(n_estimators=100, random_state=42))])

rf_model.fit(X_train, y_train_class)
gb_model.fit(X_train, y_train_class)

# Extract feature importance
rf_feature_importance = rf_model.named_steps['classifier'].feature_importances_
gb_feature_importance = gb_model.named_steps['classifier'].feature_importances_

# Create a DataFrame to hold the feature importances
feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features).tolist() + numeric_features
rf_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': rf_feature_importance})
gb_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': gb_feature_importance})

# Sort by importance
rf_importance_df = rf_importance_df.sort_values(by='Importance', ascending=False)
gb_importance_df = gb_importance_df.sort_values(by='Importance', ascending=False)

# Select top 30 features for plotting
top_n = 25
top_rf_importance_df = rf_importance_df.head(top_n)
top_gb_importance_df = gb_importance_df.head(top_n)

# Plot feature importance for Random Forest
plt.figure(figsize=(12, 6))
plt.barh(top_rf_importance_df['Feature'], top_rf_importance_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title(f'Top {top_n} Random Forest Feature Importances')
plt.gca().invert_yaxis()
plt.subplots_adjust(left=0.4)
plt.yticks(rotation=25,ha='right',)
plt.show()

# Plot feature importance for Gradient Boosting
plt.figure(figsize=(14, 5))
plt.barh(top_gb_importance_df['Feature'], top_gb_importance_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title(f'Top {top_n} Gradient Boosting Feature Importances')
plt.gca().invert_yaxis()
plt.subplots_adjust(left=0.5)
plt.yticks(rotation=25,ha='right',)
plt.show()



# Additional checks for regression assumptions
# For regression, include 'recommended' and 'stars' columns back
X_reg = data[['normalized_recommended', 'normalized_stars'] + numeric_features + categorical_features]
#y_reg = data['composite_sentiment']

# One-hot encode categorical features for regression
X_reg = pd.get_dummies(X_reg, columns=categorical_features, drop_first=True)

# Remove any remaining non-numeric columns and handle missing values
X_reg = X_reg.apply(pd.to_numeric, errors='coerce').fillna(0)

# Adding constant for intercept
X_reg = sm.add_constant(X_reg)

# Fitting the OLS regression model
model = sm.OLS(y, X_reg).fit()
# Removing variables with high p-values (>0.05) from the model, ensure at least two columns remain
high_p_value_columns = model.pvalues[model.pvalues > 0.05].index
X_reg_reduced = X_reg.drop(columns=high_p_value_columns, errors='ignore')
#print(model.summary())
# This code ensures that the constant term is always included in the regression model 
# and handles the case where all variables are removed due to high p-values. 
if 'const' not in X_reg_reduced.columns:
    X_reg_reduced = sm.add_constant(X_reg_reduced)

# Refit the model with reduced variables
final_model = sm.OLS(y, X_reg_reduced).fit()

# Outputting the final model summary
print(final_model.summary())

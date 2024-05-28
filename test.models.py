import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, precision_recall_curve
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from collections import Counter

# Load the data
file_path = 'BA_AirlineReviews_with_Sentiment2.csv'
df = pd.read_csv(file_path)

# Preprocess text data and split into training and testing sets
X = df['reviews']
y = df['SentimentScore']
y_classification = pd.cut(df['SentimentScore'], bins=[-float('inf'), 0, float('inf')],
                          labels=[0, 1]).astype('category')

# Check class distribution
print("Class distribution before splitting:", y_classification.value_counts())

# Ensure there are enough samples in each class for stratification
min_samples = 2
if any(y_classification.value_counts() < min_samples):
    raise ValueError("Some classes have fewer than 2 samples, which is too few for stratification.")

X_train, X_test, y_train, y_test = train_test_split(X, y_classification, test_size=0.30, random_state=42, stratify=y_classification)

# Use TF-IDF for feature extraction
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Use SMOTE to balance the training data
smote = SMOTE(random_state=42)
X_train_tfidf_resampled, y_train_resampled = smote.fit_resample(X_train_tfidf, y_train)

# Check the class distribution after resampling
print("Class distribution after SMOTE:", Counter(y_train_resampled))

# Define the classifiers
log_reg = LogisticRegression(max_iter=1000)
random_forest = RandomForestClassifier()
gradient_boosting = GradientBoostingClassifier()
svm = SVC(kernel='linear', probability=True)
neural_net = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)

# Hyperparameter tuning for each model using StratifiedKFold
skf = StratifiedKFold(n_splits=5)

# Example for RandomForestClassifier
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search_rf = GridSearchCV(estimator=random_forest, param_grid=param_grid_rf, scoring='precision', cv=skf, verbose=2, n_jobs=-1)
grid_search_rf.fit(X_train_tfidf_resampled, y_train_resampled)
best_rf = grid_search_rf.best_estimator_

# Train the classifiers
log_reg.fit(X_train_tfidf_resampled, y_train_resampled)
best_rf.fit(X_train_tfidf_resampled, y_train_resampled)
gradient_boosting.fit(X_train_tfidf_resampled, y_train_resampled)
svm.fit(X_train_tfidf_resampled, y_train_resampled)
neural_net.fit(X_train_tfidf_resampled, y_train_resampled)

# Make predictions
log_reg_preds = log_reg.predict(X_test_tfidf)
best_rf_preds = best_rf.predict(X_test_tfidf)
gradient_boosting_preds = gradient_boosting.predict(X_test_tfidf)
svm_preds = svm.predict(X_test_tfidf)
neural_net_preds = neural_net.predict(X_test_tfidf)

# Combine the predictions
combined_preds = pd.DataFrame({
    'Logistic Regression': log_reg_preds,
    'Random Forest': best_rf_preds,
    'Gradient Boosting': gradient_boosting_preds,
    'SVM': svm_preds,
    'Neural Network': neural_net_preds
})

# Majority voting for final prediction
final_preds = combined_preds.mode(axis=1)[0]

# Evaluation
report = classification_report(y_test, final_preds, target_names=[0, 1], zero_division=0)
print("\nClassification Report:\n", report)

# Convert categorical labels to numerical values
y_test_numerical = y_test

# Adjust decision threshold to improve precision
y_scores = best_rf.predict_proba(X_test_tfidf)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test_numerical, y_scores)

# Plot precision against threshold
plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
plt.xlabel("Threshold")
plt.legend(loc="upper left")
plt.ylim([0, 1])
plt.show()

# Set a higher threshold
threshold = 0.6
final_preds_numerical = (y_scores > threshold).astype(int)

# Convert numerical predictions back to categorical labels
final_preds_categorical = pd.Series(final_preds_numerical).map({0: 'Negative', 1: 'Positive'})

# Evaluation
report = classification_report(y_test, final_preds_categorical, target_names=['Negative', 'Positive'], zero_division=0)
print("\nClassification Report:\n", report)

# Update the DataFrame with the final predictions
df['sentimentValue'] = final_preds_categorical.values

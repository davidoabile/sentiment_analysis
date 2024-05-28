import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the data
df = pd.read_csv('BA_AirlineReviews_with_Sentiment2.csv')

# Define feature extraction
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 3), min_df=5)
X = tfidf_vectorizer.fit_transform(df['cleaned_reviews'])

# Define the target variable
y = pd.cut(df['composite_sentiment'], bins=[-float('inf'), 0.22, float('inf')],
           labels=['Negative', 'Positive']).astype('category')

# Split data for classification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

# Define models
log_reg = LogisticRegression(max_iter=1000)
random_forest = RandomForestClassifier(n_estimators=100)
gradient_boosting = GradientBoostingClassifier(n_estimators=100)
svm = SVC(kernel='linear', probability=True)

# Train the classifiers
log_reg.fit(X_train, y_train)
random_forest.fit(X_train, y_train)
gradient_boosting.fit(X_train, y_train)
svm.fit(X_train, y_train)

# Calculate log loss for training and testing data for each model
log_reg_train_loss = log_loss(y_train, log_reg.predict_proba(X_train))
log_reg_test_loss = log_loss(y_test, log_reg.predict_proba(X_test))
random_forest_train_loss = log_loss(y_train, random_forest.predict_proba(X_train))
random_forest_test_loss = log_loss(y_test, random_forest.predict_proba(X_test))
gradient_boosting_train_loss = log_loss(y_train, gradient_boosting.predict_proba(X_train))
gradient_boosting_test_loss = log_loss(y_test, gradient_boosting.predict_proba(X_test))
svm_train_loss = log_loss(y_train, svm.predict_proba(X_train))
svm_test_loss = log_loss(y_test, svm.predict_proba(X_test))

# Plot log loss for training and testing data for each model
models = ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'SVM']
train_losses = [log_reg_train_loss, random_forest_train_loss, gradient_boosting_train_loss, svm_train_loss]
test_losses = [log_reg_test_loss, random_forest_test_loss, gradient_boosting_test_loss, svm_test_loss]

plt.figure(figsize=(14, 8))
plt.bar(models, train_losses, color='blue', alpha=0.6, label='Training Loss')
plt.bar(models, test_losses, color='orange', alpha=0.6, label='Testing Loss', bottom=train_losses)
plt.xlabel('Model')
plt.ylabel('Log Loss')
plt.title('Training and Testing Log Loss for Different Models')
plt.legend()
plt.show()

print(f"Logistic Regression - Training Log Loss: {log_reg_train_loss}, Testing Log Loss: {log_reg_test_loss}")
print(f"Random Forest - Training Log Loss: {random_forest_train_loss}, Testing Log Loss: {random_forest_test_loss}")
print(f"Gradient Boosting - Training Log Loss: {gradient_boosting_train_loss}, Testing Log Loss: {gradient_boosting_test_loss}")
print(f"SVM - Training Log Loss: {svm_train_loss}, Testing Log Loss: {svm_test_loss}")

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_curve, auc, log_loss
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.calibration import CalibratedClassifierCV

stats_field = "composite_sentiment"
data = pd.read_csv('BA_AirlineReviews_with_Sentiment.csv')
y = data[stats_field]

# Define target and preprocessing
X = data.drop([stats_field, 'normalized_recommended', 'normalized_stars'], axis=1)
numeric_features = ['avg_rating_country', 'avg_rating_route', 'day_of_week', 'month']
categorical_features = ['country', 'seat_type', 'route', 'type_of_traveller', 'seat_traveller', 'country_route']

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# Model definitions with pipeline
log_reg = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', LogisticRegression(multi_class='ovr', max_iter=1000))])
random_forest = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', RandomForestClassifier(n_estimators=100))])
gradient_boosting = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('classifier', GradientBoostingClassifier(n_estimators=100))])
svm = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', SVC(kernel='rbf', class_weight='balanced', probability=True))])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Training and evaluation with loss and AUC
for name, model in {'Logistic Regression': log_reg, 'Random Forest': random_forest,
                    'Gradient Boosting': gradient_boosting, 'SVM': svm}.items():
    if name != 'SVM':  # SVM does not support partial_fit for incremental learning
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)
        print(f"{name} Loss: {log_loss(y_test, y_prob)}")

    y_pred = model.predict(X_test)
    print(f'{name} Classification Report:\n{classification_report(y_test, y_pred)}')

    # AUC and ROC
    if hasattr(model, "predict_proba"):
        fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1], pos_label='Positive')
    else:
        # CalibratedClassifierCV for models without probability estimates
        calibrated_clf = CalibratedClassifierCV(base_estimator=model, cv='prefit')
        calibrated_clf.fit(X_train, y_train)
        y_proba = calibrated_clf.predict_proba(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1], pos_label='Positive')
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc(fpr, tpr):.2f})')
    
    
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.show()

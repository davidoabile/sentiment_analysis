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
from scrape import scraper


def load_data(url = None,scrape=False, pages= 250, file_name="British_Airways_Reviews_Latest"):
    url = url or "https://www.airlinequality.com/airline-reviews/british-airways"
    return scraper(url,file_name=file_name, pages=pages) if scrape else pd.read_csv(f'{file_name}.csv')
 

def analyse(independent_variable ='composite_sentiment', scrape = False, file_name='British_Airways_Reviews_Latest', num_pages=250):
    data = load_data(scrape=scrape, file_name=file_name, pages=num_pages)
    # Load the data
    y = data[independent_variable]

    # Binning sentiment scores into categories
    y_classification = pd.cut(y, bins=[-float('inf'), 0, 2, float('inf')], labels=['Positive', 'Negative', 'Neutral'], right=True)

    # Define target and features for classification
    X = data.drop([independent_variable, 'normalized_recommended', 'normalized_stars'], axis=1)

    # Define numeric and categorical features
    numeric_features = ['avg_rating_country', 'avg_rating_route', 'day_of_week', 'month']
  

    # For regression, include 'recommended' and 'stars' columns back
    X_reg = data[['normalized_recommended', 'normalized_stars'] + numeric_features + ['seat_traveller', 'country_route', 'country_category']]
    y_reg = data[independent_variable]

    # One-hot encode categorical features for regression
    X_reg = pd.get_dummies(X_reg, columns=['seat_traveller', 'country_route', 'country_category'], drop_first=True)

    # Remove any remaining non-numeric columns and handle missing values
    X_reg = X_reg.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Adding constant for intercept
    X_reg = sm.add_constant(X_reg)

    # Fitting the initial OLS regression model
    initial_model = sm.OLS(y_reg, X_reg).fit()

    # Removing variables with high p-values (>0.05) from the model, ensure at least two columns remain
    high_p_value_columns = initial_model.pvalues[initial_model.pvalues > 0.05].index
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
    check_for_homoscedasticity(final_model)
    # Performs the Breusch-Pagan test for homoscedasticity.
    bp_test = het_breuschpagan(final_model.resid, X_reg_reduced)
    print('Breusch-Pagan test:', bp_test)

    # Q-Q plot for residuals
    sm.qqplot(final_model.resid, line='45')
    plt.title('Q-Q Plot of Residuals')
    plt.show()
    #histograms(y, independent_variable, y_classification)
    
   

def histograms(y, independent_variable ='composite_sentiment', y_classification = None):
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

    plt.xlabel(independent_variable)
    plt.ylabel('Frequency')
    plt.title('Distribution of Sentiment Score with Bins for Negative, Neutral, and Positive')
    plt.legend()
    plt.grid(False)
    plt.show()

    
    
# TODO Rename this here and in `analyse`
def check_for_homoscedasticity(final_model):
    plt.figure(figsize=(10, 6))
    plt.scatter(final_model.fittedvalues, final_model.resid)
    plt.xlabel('Fitted values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Fitted values')
    #plt.show()


if __name__ == '__main__':
    # Pass `scrape=True` to scrape the latest data
    # Pass `file_name` to specify a different file name for saving the scraped data
    analyse(scrape=False, num_pages= 300, file_name="British_Airways_Reviews_Latest", independent_variable='SentimentScore')
    

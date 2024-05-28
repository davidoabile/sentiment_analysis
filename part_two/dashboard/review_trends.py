import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('British_Airways_Reviews_Latest.csv')

# Combine country and route for grouping
df['country_route'] = df['country'] + ' - ' + df['route']

# Calculate the average sentiment score for each country-route combination
country_route_sentiment = df.groupby('country_route')['SentimentScore'].mean().reset_index()

# Sort the country-route combinations by average sentiment score
country_route_sentiment = country_route_sentiment.sort_values(by='SentimentScore', ascending=False)

# Select the top and bottom N routes
N = 10
#top_bottom_routes = pd.concat([country_route_sentiment.head(N), country_route_sentiment.tail(N)])
#top_bottom_routes = country_route_sentiment.head(N)

# Plot settings
sns.set(style="white")
plt.figure(figsize=(16, 18))

# Plot the average sentiment score for selected country-route combinations
sns.barplot(x='SentimentScore', y='country_route', data=country_route_sentiment.head(N), palette='viridis')

# Add title and labels
plt.title('Positive Average Customer Sentiment by Country and Route')
plt.xlabel('Average Sentiment Score')
plt.ylabel('Country - Route')
plt.xlim(0, 1 )  # Sentiment score ranges from -1 to 1
plt.xticks(rotation=25, ha='right') 
plt.yticks(rotation=25,ha='right',)
plt.subplots_adjust(left=0.3)

plt.show()

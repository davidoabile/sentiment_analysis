"""
To visualize review trends, we can create several types of plots, such as:
Sentiment analysis is the process of determining and categorizing opinions expressed in a piece of text, especially to determine the writer's attitude towards a particular topic. This is typically done by analyzing the text's emotional tone. 
Sentiment Distribution: Visualize the distribution of sentiments (positive, negative, neutral) over time.
Review Count Over Time: Show the number of reviews received over time (by day, month, or year).
Average Rating Over Time: Display the average star rating over time.
Category Distribution: Show the distribution of review categories (Customer Service, Refund, etc.) over time.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('British_Airways_Reviews_Latest.csv')

# Display the first few rows of the dataframe to ensure it's loaded correctly
print(df.head())

# Ensure the 'year' column is in integer format
df['year'] = df['year'].astype(int)

# Plot settings
sns.set(style="white")
plt.figure(figsize=(14, 8))

# Sentiment Distribution Over Time
sentiment_counts = df.groupby(['year', 'composite_sentiment']).size().unstack().fillna(0)

# Plotting the sentiment distribution over time
sentiment_counts.plot(kind='line',  stacked=True, legend=False)
plt.title('Sentiment Distribution Over Time')
plt.xlabel('Year')
plt.ylabel('Number of Reviews')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()



# Calculate the number of complaints by category for each route
route_issues = df.groupby(['route', 'Category']).size().unstack(fill_value=0)

# Identify the top 3 routes with the most negative feedback
top_negative_routes = route_sentiment.sort_values(by='SentimentScore').head(3)['route'].tolist()

# Filter complaints for these routes
top_route_issues = route_issues.loc[top_negative_routes]
print(top_route_issues)



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('/mnt/data/BA_AirlineReviews_with_Sentiment.csv')

# Group by route and Category to get the count of each category for each route
category_distribution_by_route = df.groupby(['route', 'Category']).size().reset_index(name='Count')

# For better visualization, sort the routes within each category by count
category_distribution_by_route = category_distribution_by_route.sort_values(by=['Category', 'Count'], ascending=[True, False])

# Plot settings
sns.set(style="whitegrid")
plt.figure(figsize=(14, 10))

# Plot the complaints for each route
sns.barplot(x='Count', y='route', hue='Category', data=category_distribution_by_route, palette='viridis')

# Add title and labels
plt.title('Complaints Distribution Across Routes')
plt.xlabel('Number of Reviews')
plt.ylabel('Route')

# Adjust legend
plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.show()



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('/mnt/data/BA_AirlineReviews_with_Sentiment.csv')

# Create a new column for sentiment category based on the SentimentScore
df['SentimentCategory'] = df['SentimentScore'].apply(lambda x: 'Positive' if x > 0 else 'Negative' if x < 0 else 'Neutral')

# Group by type_of_traveller and Category to get the count of each category for each customer type
category_distribution = df.groupby(['type_of_traveller', 'Category']).size().reset_index(name='Count')

# For better visualization, sort the categories within each customer type by count
category_distribution = category_distribution.sort_values(by=['type_of_traveller', 'Count'], ascending=[True, False])

# Plot settings
sns.set(style="whitegrid")
plt.figure(figsize=(14, 10))

# Plot the top complaints for each customer type
sns.barplot(x='Count', y='Category', hue='type_of_traveller', data=category_distribution, palette='viridis')

# Add title and labels
plt.title('Top Complaints by Customer Type')
plt.xlabel('Number of Reviews')
plt.ylabel('Category')

# Adjust legend
plt.legend(title='Customer Type', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.show()



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('/mnt/data/BA_AirlineReviews_with_Sentiment.csv')

# Calculate the average sentiment score for each route
route_sentiment = df.groupby('route')['SentimentScore'].mean().reset_index()

# Sort the routes by average sentiment score in ascending order to get the most negative routes
route_sentiment = route_sentiment.sort_values(by='SentimentScore', ascending=True)

# Select the top 10 routes with the lowest average sentiment scores
most_negative_routes = route_sentiment.head(10)

# Plot settings
sns.set(style="whitegrid")
plt.figure(figsize=(14, 8))

# Plot the average sentiment score for the most negative routes
sns.barplot(x='SentimentScore', y='route', data=most_negative_routes, palette='viridis')

# Add title and labels
plt.title('Top 10 Routes with Most Negative Sentiment')
plt.xlabel('Average Sentiment Score')
plt.ylabel('Route')
plt.xlim(-1, 1)  # Sentiment score ranges from -1 to 1

plt.show()


#Analyze sentiment distribution by customer type.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('/mnt/data/BA_AirlineReviews_with_Sentiment.csv')

# Create a new column for sentiment category based on the SentimentScore
df['SentimentCategory'] = df['SentimentScore'].apply(lambda x: 'Positive' if x > 0 else 'Negative' if x < 0 else 'Neutral')

# Group by type_of_traveller and SentimentCategory to get the count of each sentiment for each customer type
sentiment_distribution = df.groupby(['type_of_traveller', 'SentimentCategory']).size().unstack(fill_value=0).reset_index()

# Melt the DataFrame for easier plotting
sentiment_distribution = sentiment_distribution.melt(id_vars='type_of_traveller', value_vars=['Positive', 'Negative', 'Neutral'],
                                                     var_name='SentimentCategory', value_name='Count')

# Plot settings
sns.set(style="whitegrid")
plt.figure(figsize=(14, 8))

# Plot the sentiment distribution by customer type
sns.barplot(x='type_of_traveller', y='Count', hue='SentimentCategory', data=sentiment_distribution, palette='viridis')

# Add title and labels
plt.title('Sentiment Distribution by Customer Type')
plt.xlabel('Customer Type')
plt.ylabel('Count of Reviews')

plt.show()




#Analyze top routes with best sentiment.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('/mnt/data/BA_AirlineReviews_with_Sentiment.csv')

# Calculate the average sentiment score for each route
route_sentiment = df.groupby('route')['SentimentScore'].mean().reset_index()

# Sort the routes by average sentiment score in descending order
route_sentiment = route_sentiment.sort_values(by='SentimentScore', ascending=False)

# Select the top 10 routes with the highest average sentiment scores
top_routes = route_sentiment.head(10)

# Plot settings
sns.set(style="whitegrid")
plt.figure(figsize=(14, 8))

# Plot the average sentiment score for the top routes
sns.barplot(x='SentimentScore', y='route', data=top_routes, palette='viridis')

# Add title and labels
plt.title('Top 10 Routes with Best Sentiment')
plt.xlabel('Average Sentiment Score')
plt.ylabel('Route')
plt.xlim(-1, 1)  # Sentiment score ranges from -1 to 1

plt.show()


#Analyze average sentiment for each year.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('/mnt/data/BA_AirlineReviews_with_Sentiment.csv')

# Convert the date column to datetime
df['date'] = pd.to_datetime(df['date'])

# Extract the year from the date column
df['year'] = df['date'].dt.year

# Calculate the average sentiment score for each year
yearly_sentiment = df.groupby('year')['SentimentScore'].mean().reset_index()

# Plot settings
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))

# Plot the average sentiment score for each year
sns.barplot(x='year', y='SentimentScore', data=yearly_sentiment, palette='viridis')

# Add title and labels
plt.title('Average Customer Sentiment by Year')
plt.xlabel('Year')
plt.ylabel('Average Sentiment Score')
plt.ylim(-1, 1)  # Sentiment score ranges from -1 to 1

plt.show()


# Convert the date column to datetime
df['date'] = pd.to_datetime(df['date'])

# Set the date as the index
df.set_index('date', inplace=True)

# Calculate the average sentiment score for each route over time
route_sentiment_trends = df.groupby([pd.Grouper(freq='M'), 'route'])['SentimentScore'].mean().unstack()

# Plot settings
sns.set(style="whitegrid")
plt.figure(figsize=(14, 8))

# Plot the sentiment trends for each route
route_sentiment_trends.plot(ax=plt.gca(), linewidth=2.5)

# Add title and labels
plt.title('Sentiment Trends by Route Over Time')
plt.xlabel('Date')
plt.ylabel('Average Sentiment Score')
plt.legend(title='Route', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)

plt.show()



#If you want to analyze sentiment trends over time for each route, you can modify the code as follows:

# Convert the date column to datetime
df['date'] = pd.to_datetime(df['date'])

# Set the date as the index
df.set_index('date', inplace=True)

# Calculate the average sentiment score for each route over time
route_sentiment_trends = df.groupby([pd.Grouper(freq='M'), 'route'])['SentimentScore'].mean().unstack()

# Plot settings
sns.set(style="whitegrid")
plt.figure(figsize=(14, 8))

# Plot the sentiment trends for each route
route_sentiment_trends.plot(ax=plt.gca(), linewidth=2.5)

# Add title and labels
plt.title('Sentiment Trends by Route Over Time')
plt.xlabel('Date')
plt.ylabel('Average Sentiment Score')
plt.legend(title='Route', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)

plt.show()



#Analyze customer sentiment by route.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('/mnt/data/BA_AirlineReviews_with_Sentiment.csv')

# Calculate the average sentiment score for each route
route_sentiment = df.groupby('route')['SentimentScore'].mean().reset_index()

# Sort the routes by average sentiment score for better visualization
route_sentiment = route_sentiment.sort_values(by='SentimentScore', ascending=False)

# Plot settings
sns.set(style="whitegrid")
plt.figure(figsize=(14, 8))

# Plot the average sentiment score for each route
sns.barplot(x='SentimentScore', y='route', data=route_sentiment, palette='viridis')

# Add title and labels
plt.title('Average Customer Sentiment by Route')
plt.xlabel('Average Sentiment Score')
plt.ylabel('Route')
plt.xlim(-1, 1)  # Sentiment score ranges from -1 to 1

plt.show()


#How can we visualize review trends?
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('/mnt/data/BA_AirlineReviews_with_Sentiment.csv')

# Convert the date column to datetime
df['date'] = pd.to_datetime(df['date'])

# Plot settings
sns.set(style="whitegrid")
plt.figure(figsize=(14, 8))

# Sentiment Distribution Over Time
plt.subplot(2, 2, 1)
sentiment_counts = df.groupby([df['date'].dt.to_period('M'), 'SentimentScoreValue']).size().unstack().fillna(0)
sentiment_counts.plot(kind='bar', stacked=True, ax=plt.gca())
plt.title('Sentiment Distribution Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Reviews')
plt.xticks(rotation=45)

# Review Count Over Time
plt.subplot(2, 2, 2)
df.set_index('date').resample('M').size().plot(ax=plt.gca())
plt.title('Number of Reviews Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Reviews')
plt.xticks(rotation=45)

# Average Rating Over Time
plt.subplot(2, 2, 3)
df.set_index('date').resample('M')['stars'].mean().plot(ax=plt.gca())
plt.title('Average Rating Over Time')
plt.xlabel('Date')
plt.ylabel('Average Rating')
plt.ylim(0, 5)
plt.xticks(rotation=45)

# Category Distribution Over Time
plt.subplot(2, 2, 4)
category_counts = df.groupby([df['date'].dt.to_period('M'), 'Category']).size().unstack().fillna(0)
category_counts.plot(kind='bar', stacked=True, ax=plt.gca())
plt.title('Category Distribution Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Reviews')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


#Plot average rating for categories.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('/mnt/data/BA_AirlineReviews_with_Sentiment.csv')

# Convert the date column to datetime
df['date'] = pd.to_datetime(df['date'])

# Calculate the average rating for each category
category_avg_rating = df.groupby('Category')['stars'].mean().reset_index()

# Plot settings
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))

# Plot the average rating for each category
sns.barplot(x='Category', y='stars', data=category_avg_rating, palette='viridis')

# Add title and labels
plt.title('Average Rating for Each Category')
plt.xlabel('Category')
plt.ylabel('Average Rating')
plt.ylim(0, 5)
plt.xticks(rotation=45)

plt.show()


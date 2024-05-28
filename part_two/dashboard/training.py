import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('BA_AirlineReviews_with_Sentiment.csv')

# Convert the date column to datetime
df['date'] = pd.to_datetime(df['date'])

# Calculate the average rating for each category
category_avg_rating = df.groupby('Category')['SentimentScore'].mean().reset_index()

# Plot settings
sns.set(style="white")
plt.figure(figsize=(10, 6))

# Plot the average rating for each category
sns.barplot(x='Category', y='SentimentScore', data=category_avg_rating, palette='viridis')

# Add title and labels
plt.title('Average Sentiment for Each Category')
plt.xlabel('Category')
plt.ylabel('Average Sentiment Score')
plt.ylim(0, 0.15)  # Adjusted to fit the normalized data range

# Rotate and align x labels
plt.xticks(rotation=25, ha='right')  # Rotate x-axis labels and align them to the right
plt.yticks([])
plt.show()

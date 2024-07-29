#!/usr/bin/env python
# coding: utf-8

# ### This code supports the analysis of negative experiences reported in Apple App Store user feedback on the Wysa chatbot from 2022, 2023, and January to May 2024. The review was conducted for an MS IT8010 IT Capstone Project at the University of Cincinnati, School of Information Technology.
# #### Student: Linda Sheppler
# #### Professor: Dr. Murat Ozer
# #### Date: July 17, 2024

# In[58]:


# Install Apple app-store-scraper
# Create separate CSV files 2022, 2023, Jan-May 2024

from app_store_scraper import AppStore
from pprint import pprint
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = AppStore(country="us", app_name="wysa-mental-health-support")
app.review(how_many=200)

# Change the dates to output 2022, 2023, or Jan-May 2024, including CSV filenames
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 5, 31)

filtered_reviews = []

for review in app.reviews:
    review_date = review['date']
    if start_date <= review_date <= end_date:
        # Exclude the username
        review.pop('username', None)
        filtered_reviews.append(review)

# Analyze the reviews
analyzer = SentimentIntensityAnalyzer()

for review in filtered_reviews:
    review_text = review['review']
    sentiment = analyzer.polarity_scores(review_text)
    print(f"Review: {review_text}\nSentiment: {sentiment}\n")


# In[59]:


# Clean and print text, guidance from https://stackoverflow.com/questions/47081682/removing-punctuation-and-change-to-lowercase-in-python-csv-file

import string

# Read the CSV file and convert lines to lowercase
with open('Apple_reviews_sentiment_2024.csv') as csvfile:
    lines = [line.lower() for line in csvfile]

# Remove punctuation
punc_table = str.maketrans({key: None for key in string.punctuation})

new_res = str(lines).translate(punc_table)

print(new_res)


# In[60]:


import csv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

# Open the CSV file in write mode
with open('Apple_reviews_sentiment_2024.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(["Review", "Positive", "Neutral", "Negative", "Compound"])

    for review in filtered_reviews:
        review_text = review['review']
        sentiment = analyzer.polarity_scores(review_text)
        # Write to the CSV file
        writer.writerow([review_text, sentiment['pos'], sentiment['neu'], sentiment['neg']])


# In[61]:


import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
#import pyplot subpackage from matplotlib as plt
from matplotlib import pyplot as plt
import numpy as np
#use %matplotlib inline with Jupyter notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# Add the following codes to avoid INFO message
import os
# Set the threads to 5
os.environ['NUMEXPR_NUM_THREADS'] = '5'
# Import numexpr
import numexpr as ne


# In[62]:


# Select the CSV file that matches the start/end dates
df = pd.read_csv('Apple_reviews_sentiment_2024.csv')


# In[63]:


sns.histplot(data=df, x="Negative", color="red", label="Negative review", alpha=0.8, stat="density")
sns.histplot(data=df, x="Positive", color="green", label="Positive review", alpha=0.4, stat="density")

plt.xlabel("Sentiment")
plt.ylabel("Frequency")
# Change the title to match the period of the data
plt.title("Wysa: Apple App Store Reviews Jan-May 2024")

plt.legend()
plt.show()


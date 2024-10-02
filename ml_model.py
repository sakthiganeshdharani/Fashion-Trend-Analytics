import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import string
import regex as re
import pandas as pd
from flask import Flask, request, render_template
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import numpy as np
from pymongo import MongoClient
import warnings
warnings.filterwarnings("ignore")

MONGODB_HOST = "localhost"
MONGODB_PORT = 27017
MONGODB_DATABASE = "FashionStore"
MONGODB_COLLECTION = "Data"

client = MongoClient(MONGODB_HOST, MONGODB_PORT)
db = client[MONGODB_DATABASE]
collection = db[MONGODB_COLLECTION]

def run_ml_model():
    def checkscore(weights):
        # Make predictions using the trained regressors
        model1_predictions = lr.predict(X_test)
        model2_predictions = clf.predict(X_test)

        # Combine predictions using weighted voting
        final_prediction = weights[0] * model1_predictions + weights[1] * model2_predictions

        # Calculate the MSE
        ensemble_mse = mean_squared_error(y_test, final_prediction)
        return ensemble_mse
    def preprocess_tags(row):
        tokens = word_tokenize(row)

        # Remove punctuation and convert to lowercase
        tokens = [word.lower() for word in tokens if word.isalnum()]

        # Get NLTK English stop words
        stop_words = set(stopwords.words('english'))

        # Remove stop words
        filtered_tokens = [word for word in tokens if word not in stop_words]
        
        return ','.join(filtered_tokens)
    base_url = "https://www.vogue.in/fashion/fashion-trends?page="
    page_number = 1  # Start with the first page
    text=''
    while page_number<5:
        url = base_url + str(page_number)
        response = requests.get(url)

        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')

            # Scraping logic here - extract the content you need from the current page
            anchor_tags = soup.find_all('a')
        
            fashion_related_content = []

            # Iterate through anchor tags and extract the content within them
            for anchor in anchor_tags:
                content = anchor.get_text().strip()  # Get the text content and remove extra spaces
                fashion_related_content.append(content)

            # Define a function to count words in a sentence
            def count_words(sentence):
                words = re.findall(r'\b\w+\b', sentence)
                return len(words)

            # Print sentences with more than 10 words
            for content in fashion_related_content:
                if count_words(content) > 10:
                    text=text+''+content

            # Check for a "Next" button or other pagination indicator
            next_button = soup.find('a', text='Next Page')
            if next_button:
                page_number += 1
            else:
                break  # No more pages to scrape, exit the loop
        else:
            print("Failed to retrieve the webpage. Status code:", response.status_code)
            break
    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove punctuation and convert to lowercase
    tokens = [word.lower() for word in tokens if word.isalnum()]


    stop_words = set(stopwords.words('english'))


    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Count keyword frequency
    keyword_frequency = Counter(filtered_tokens)
    ranked_keywords = sorted(keyword_frequency.items(), key=lambda x: x[1], reverse=True)
    
    data = list(collection.find({}))

    df = pd.DataFrame(data)

    df=df.dropna(axis=0)

    df['tags'] = df['tags'].apply(preprocess_tags)

    keyword_frequency_dict=dict(keyword_frequency)
    prokeyfreq=[]
    
    for tag in df['tags']:
            # Split the input string by commas and remove leading/trailing spaces
        words = [word.strip() for word in tag.split(",")]

        # Initialize a variable to store the sum
        total_sum = 0

        # Iterate through the words and sum their values from the dictionary
        for word in words:
            if word in keyword_frequency_dict:
                total_sum += keyword_frequency_dict[word]

        # Print the total sum
        prokeyfreq.append(total_sum)
    df['Keyword Freq']=prokeyfreq
    
    data = list(collection.find({}))

# Convert the retrieved data to a DataFrame
    d = pd.DataFrame(data)

    d=d.dropna(axis=0)
    np.random.seed(42)

    lb=preprocessing.LabelEncoder()
    d['product_color']=lb.fit_transform(d['product_color'])
    d['product_variation_size_id']=lb.fit_transform(d['product_variation_size_id'])
    d['shipping_option_name']=lb.fit_transform(d['shipping_option_name'])
    d['crawl_month'] = pd.to_datetime(d['crawl_month'])
    reference_date = pd.Timestamp("1970-01-01")
    d['timestamp'] = (d['crawl_month'] - reference_date).dt.total_seconds()

    X = d[['price', 'retail_price',  'uses_ad_boosts', 'rating', 'rating_count', 'rating_five_count', 'rating_four_count', 'rating_three_count', 'rating_two_count', 'rating_one_count', 'badge_local_product', 'badge_product_quality', 'badge_fast_shipping', 'shipping_option_price', 'shipping_is_express', 'inventory_total', 'merchant_rating', 'product_color', 'product_variation_size_id', 'shipping_option_name','timestamp']]
    y = d['units_sold']

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)   

    clf=RandomForestRegressor(n_estimators=100,max_depth=5)
    lr=LinearRegression()
    
    clf.fit(X_train, y_train)
    lr.fit(X_train, y_train)
    
    bounds = ((0, None), (0, None))  
    initial_guess=[0.5,0.5]
    result = minimize(checkscore, method='Nelder-Mead',x0=initial_guess,bounds=bounds)
    optimal_weights = result.x

    final_prediction = optimal_weights[0] * lr.predict(X_test) + optimal_weights[1] * clf.predict(X_test)
    rand_score=r2_score(y_test, final_prediction)
    err=mean_squared_error(y_test, final_prediction)
    demand=[]
    inp = '2023-09-28'
    temp = pd.to_datetime(inp)
    reference_date = pd.Timestamp("1970-01-01")
    x = (temp - reference_date).total_seconds()
    for index, row in d.iterrows():
        test=[[row['price'], row['retail_price'],  row['uses_ad_boosts'], row['rating'], row['rating_count'], row['rating_five_count'], row['rating_four_count'], row['rating_three_count'], row['rating_two_count'], row['rating_one_count'], row['badge_local_product'], row['badge_product_quality'], row['badge_fast_shipping'],row[ 'shipping_option_price'],row[ 'shipping_is_express'], row['inventory_total'],row[ 'merchant_rating'], row['product_color'], row['product_variation_size_id'], row['shipping_option_name'],x]]
        final_prediction = optimal_weights[0] * lr.predict(test) + optimal_weights[1] * clf.predict(test)
        demand.append(final_prediction)
    df['demand']=demand
    df['demand'] = df['demand'].apply(lambda x: x[0])
    perfmetrics=[]

    for sales, key in zip(df['demand'], df['Keyword Freq']):
        perfmetrics.append(sales * key)

    df['Performance Metrics']=perfmetrics
    df['Performance Metrics'] = df['Performance Metrics'].apply(lambda x: x[0] if isinstance(x, list) else x)
    df['rank'] = df['Performance Metrics'].rank(ascending=False, method='dense')
    updated_data = df.to_dict(orient='records')
    collection.delete_many({})
    collection.insert_many(updated_data)
    return 
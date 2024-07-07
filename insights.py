import pandas as pd
import json
import re
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer # type: ignore
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import seaborn as sns


# Clean the text data
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.lower()  # Convert to lowercase
    return text








# Load the chat history JSON file
with open('./data/chat_history.json', 'r') as file:
    chat_data = json.load(file)

# Adding cleaned text to chat_data 
for customer in chat_data:
    for message in customer['chat_history']:
        if 'message' in message:
            message['cleaned_text'] = clean_text(message['message'])
        elif 'agent_message' in message:
            message['cleaned_text'] = clean_text(message['agent_message'])
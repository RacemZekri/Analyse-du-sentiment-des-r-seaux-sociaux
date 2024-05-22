import pandas as pd
import re
import plotly.express as px
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

nltk.download('stopwords')
nltk.download('wordnet')

df = pd.read_csv('tweets.csv')


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_tweet(tweet):
    # Remove mentions, hashtags, URLs, and special characters
    tweet = re.sub(r"@\w+|#\w+|http\S+|[^a-zA-Z\s]", "", tweet)
    tweet = tweet.lower()
    tokens = tweet.split()
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return " ".join(tokens)


df['cleaned_tweet'] = df['tweet'].apply(preprocess_tweet)

# Sentiment analysis function
def get_sentiment(tweet):
    analysis = TextBlob(tweet)
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity == 0:
        return 'neutral'
    else:
        return 'negative'

# Sentiment analysis
df['sentiment'] = df['cleaned_tweet'].apply(get_sentiment)

print(df[['tweet', 'cleaned_tweet', 'sentiment']])

#
fig = px.histogram(df, x='sentiment', color='sentiment', barmode='group')
fig.update_layout(title='Histogramme des sentiments', xaxis_title='Sentiment', yaxis_title='Nombre de tweets')

fig.show()





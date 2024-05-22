import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load the dataset
df = pd.read_csv('tweets.csv')

# Preprocessing function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_tweet(tweet):
    # Remove mentions, hashtags, URLs, and special characters
    tweet = re.sub(r"@\w+|#\w+|http\S+|[^a-zA-Z\s]", "", tweet)
    # Convert to lowercase
    tweet = tweet.lower()
    # Tokenize and remove stopwords and lemmatize
    tokens = tweet.split()
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return " ".join(tokens)

# Apply preprocessing
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

# Apply sentiment analysis
df['sentiment'] = df['cleaned_tweet'].apply(get_sentiment)

# Display the results
print(df[['tweet', 'cleaned_tweet', 'sentiment']])

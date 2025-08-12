import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

def load_data(filepath):
    data = pd.read_csv(filepath)
    print(f"Data shape: {data.shape}")
    return data

def preprocess_data(df):
    # Drop NaNs or empty tweets
    df = df.dropna(subset=['tweet_text', 'sentiment'])
    return df

def main():
    data_path = '../data/twitter_sentiment.csv'  # Adjust your path
    df = load_data(data_path)
    df = preprocess_data(df)

    X = df['tweet_text']
    y = df['sentiment']

    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred))

    # Save vectorizer and model
    joblib.dump(model, '../models/sentiment_model.pkl')
    joblib.dump(vectorizer, '../models/vectorizer.pkl')
    print("Model and vectorizer saved.")

if __name__ == '__main__':
    main()


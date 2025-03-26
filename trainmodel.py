import re
import string
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

def clean_text(text: str) -> str:
    """Preprocesses text by converting to lowercase and removing punctuation."""
    return re.sub(f"[{string.punctuation}]", "", str(text).lower())

def load_and_preprocess_data(csv_path: str):
    """Loads dataset, preprocesses text, and encodes labels."""
    df = pd.read_csv(csv_path)
    if not {'Review', 'Recommended'}.issubset(df.columns):
        raise ValueError("CSV must contain 'Review' and 'Recommended' columns")
    
    df = df[['Review', 'Recommended']].dropna()
    df['Review'] = df['Review'].apply(clean_text)
    df['Recommended'] = df['Recommended'].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)
    
    return df

def train_model(X_train, y_train):
    """Trains a logistic regression model."""
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluates model performance and prints metrics."""
    predictions = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))

def main(csv_path: str, model_path: str, vectorizer_path: str):
    """Main function to execute training workflow."""
    df = load_and_preprocess_data(csv_path)
    
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['Review'])
    y = df['Recommended'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    
    evaluate_model(model, X_test, y_test)
    
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Model saved as '{model_path}', vectorizer saved as '{vectorizer_path}'")

if __name__ == "__main__":
    main("AirlineReviews.csv", "sentiment_model.pkl", "vectorizer.pkl")

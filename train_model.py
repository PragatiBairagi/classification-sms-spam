import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load your dataset
df = pd.read_csv('spam.csv', encoding='latin-1')  # Update the path to your dataset if it's in a different location
df = df[['v1', 'v2']]
df.columns = ['target', 'text']

# Convert target to numerical values
df['target'] = df['target'].map({'ham': 0, 'spam': 1})

# Split data into features and target
X = df['text']
y = df['target']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=2500)  # Adjust max_features as needed
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Initialize and train the model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the vectorizer and model
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model and vectorizer saved successfully!")

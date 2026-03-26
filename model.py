import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("="*50)
print("FAKE REVIEW DETECTION PROJECT")
print("="*50)

print("\nLoading dataset...")
df = pd.read_csv('fake reviews dataset.csv')
print(f"Loaded {len(df)} reviews")

print("\nColumns in dataset:")
print(df.columns.tolist())

print("\nFirst 2 rows:")
print(df.head(2))

print("\nConverting text to numbers...")
vectorizer = CountVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['text_'].fillna(''))
print(f"Created {X.shape[1]} features")

print("\nPreparing labels...")
y = df['label']
print(f"Labels found: {y.unique().tolist()}")

print("\nSplitting data into train and test...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training: {X_train.shape[0]} reviews")
print(f"Testing: {X_test.shape[0]} reviews")

print("\nTraining Naive Bayes model...")
model = MultinomialNB()
model.fit(X_train, y_train)
print("Training complete")

print("\nMaking predictions...")
y_pred = model.predict(X_test)

print("\n" + "="*50)
print("RESULTS")
print("="*50)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nACCURACY: {accuracy:.2%}")

print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred))

print("\nCONFUSION MATRIX:")
print(confusion_matrix(y_test, y_pred))

print("\n" + "="*50)
print("TOP 15 WORDS THAT INDICATE FAKE REVIEWS")
print("="*50)

feature_names = vectorizer.get_feature_names_out()
top_features = np.argsort(model.feature_log_prob_[1])[-15:]

for i, idx in enumerate(reversed(top_features), 1):
    print(f"{i:2d}. {feature_names[idx]}")

print("\n" + "="*50)
print("PROJECT COMPLETE!")
print("="*50)

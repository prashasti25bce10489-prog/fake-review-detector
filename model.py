import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("="*50)
print("okay lets see if these reviews are real or just paid")
print("="*50)

file_name= "fake reviews dataset.csv"

try:
    df = pd.read_csv(file_name)
    print(f"\nCool, found {len(df)} reviews in the file")
except:
    print(f"\nOops, can't find {file_name}. Did you move it?")
    exit()
print("\nQuick look at what columns we got:")
for col in df.columns:
    print(f"   - {col}")


print("\nFirst couple of reviews just to see what we're dealing with:")
for i in range(min(2, len(df))):
    print(f"\nReview #{i+1}:")
    print(f"   Said: {df['text_'].iloc[i][:120]}...")
    print(f"   Marked as: {df['label'].iloc[i]}")

print("\nConverting text to numbers...")
vectorizer = CountVectorizer(max_features=1000)

review_texts = df['text_'].fillna('')
X = vectorizer.fit_transform(review_texts)

y = df['label']

print(f"Done. Now we have {X.shape[1]} features from {X.shape[0]} reviews")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining on {X_train.shape[0]} reviews")
print(f"Testing on {X_test.shape[0]} reviews")

print("\nTraining the model...")
model = MultinomialNB()
model.fit(X_train, y_train)
print("Done training")

print("\nRunning predictions...")
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print("\n" + "-"*40)
print(f"Accuracy: {acc:.2%}")
print("-"*40)

print("\nDetailed breakdown:")
print(classification_report(y_test, y_pred))

print("\nConfusion matrix (actual vs predicted):")
print(confusion_matrix(y_test, y_pred))

print("\n" + "-"*40)
print("Words that scream FAKE:")
print("-"*40)

feature_names = vectorizer.get_feature_names_out()
fake_scores = model.feature_log_prob_[1]

top_words = np.argsort(fake_scores)[-15:]

for i, idx in enumerate(reversed(top_words), 1):
    print(f"{i:2d}. {feature_names[idx]}")

print("\nMakes sense. Fake reviews love extreme words.")
print("Real reviews are more chill and specific.")

print("\n" + "-"*40)
print("That's it. Done.")
print("-"*40)
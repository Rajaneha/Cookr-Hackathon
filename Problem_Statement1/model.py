from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from sklearn.metrics import accuracy_score
import streamlit as st
from catboost import CatBoostClassifier
import pandas as pd
import joblib

df = pd.read_csv('IndianFoodDatasetCSV.csv')
df.dropna(inplace=True)

df.drop(['RecipeName','Ingredients','TranslatedIngredients','PrepTimeInMins','CookTimeInMins','TotalTimeInMins','Servings', 'Instructions','TranslatedInstructions','URL'], axis=1,inplace=True)
    
# Split data into features and labels
X = df['TranslatedRecipeName']
y = df[['Cuisine', 'Course', 'Diet']]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Text vectorization using TF-IDF
tfidf = TfidfVectorizer(max_features=1000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train separate CatBoost classifiers for each label
classifiers = {}
for label in y_train.columns:
    classifier = CatBoostClassifier(iterations=100, verbose=False)
    classifier.fit(X_train_tfidf, y_train[label])
    classifiers[label] = classifier
print(classifiers)
# Predictions
y_pred = {}
for label, classifier in classifiers.items():
    y_pred[label] = classifier.predict(X_test_tfidf)

# Print classification report
print("Classification Report:")
for label in y_test.columns:
    print(f"\n{label}:")
    print(classification_report(y_test[label], y_pred[label]))

# Calculate accuracy for each label
accuracies = {}
for label in y_test.columns:
    accuracies[label] = accuracy_score(y_test[label], y_pred[label])

# Calculate average accuracy
average_accuracy = sum(accuracies.values()) / len(accuracies)

# Print accuracy for each label
print("\nAccuracy for Each Label:")
for label, accuracy in accuracies.items():
    print(f"{label}: {accuracy}")
# Print average accuracy
print("\nAverage Accuracy:", average_accuracy)

joblib.dump(classifiers, 'model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
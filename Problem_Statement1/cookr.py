import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from sklearn.metrics import accuracy_score
import streamlit as st
from catboost import CatBoostClassifier
import pickle
import joblib
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
import soundex

def soundex_generator(token):
   
    # Convert the word to upper 
    # case for uniformity
    token = token.upper()
 
    soundex = ""
 
    # Retain the First Letter
    soundex += token[0]
 
    # Create a dictionary which maps 
    # letters to respective soundex
    # codes. Vowels and 'H', 'W' and
    # 'Y' will be represented by '.'
    dictionary = {"BFPV": "1", "CGJKQSXZ": "2",
                  "DT": "3",
                  "L": "4", "MN": "5", "R": "6",
                  "AEIOUHWY": "."}
 
    # Enode as per the dictionary
    for char in token[1:]:
        for key in dictionary.keys():
            if char in key:
                code = dictionary[key]
                if code != '.':
                    if code != soundex[-1]:
                        soundex += code
 
    # Trim or Pad to make Soundex a
    # 4-character code
    soundex = soundex[:7].ljust(7, "0")
 
    return soundex

# Add this function to your code
def is_similar_word(word1, word2):
    return soundex_generator(word1) == soundex_generator(word2)

# Initialize the Porter Stemmer
stemmer = PorterStemmer()

# Function to preprocess and normalize input text
def preprocess_input(input_text):
    # Tokenize the input text
    tokens = word_tokenize(input_text)
    # Apply stemming to each token
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    # Join the stemmed tokens back into a single string
    preprocessed_text = ' '.join(stemmed_tokens)
    return preprocessed_text.lower()  # Normalize to lowercase

# Streamlit UI
def Add_Item():
    with open('model.pkl', 'rb') as model_file:
        model = joblib.load(model_file)
        
    with open('tfidf_vectorizer.pkl','rb') as model_file1:
        tfidf_vector = joblib.load(model_file1)
        
    input_recipe = st.text_input("Enter the recipe name: ")
    preprocessed_input = preprocess_input(input_recipe)
    
    if st.button('Add'):
        # Check if the CSV file exists
        file_path = 'recipes_data.csv'
        file_exists = os.path.isfile(file_path)

        # Load existing data from CSV
        if file_exists:
            existing_data = pd.read_csv(file_path)
            
            # Check if the normalized recipe or a similar word already exists
            if any(existing_data['RecipeName'].apply(lambda x: is_similar_word(x.lower(), preprocessed_input))):
                st.write("Recipe or similar word already exists in the Kitchen.")
            else:
                # Make predictions using the trained model
                input_tfidf = tfidf_vector.transform([preprocessed_input])
                predicted_labels = {}
                for label, classifier in model.items():
                    predicted_labels[label] = classifier.predict(input_tfidf)

                st.write("Predicted Labels:")
                for label, prediction in predicted_labels.items():
                    st.write(f"{label}: {prediction[0]}")

                # Add new data to DataFrame
                new_data = {
                    'RecipeName': [input_recipe],
                    'Cuisine': [predicted_labels['Cuisine'][0]],
                    'Course': [predicted_labels['Course'][0]],
                    'Diet': [predicted_labels['Diet'][0]]
                }

                # Append to existing DataFrame
                df = existing_data.append(pd.DataFrame(new_data), ignore_index=True)

                # Write to CSV with or without header
                df.to_csv('recipes_data.csv', mode='a', header=not file_exists, index=False)

      
def Search():
    st.title("Search for Food")

    # Read the recipes_data.csv file
    try:
        recipes_data = pd.read_csv('recipes_data.csv')
    except FileNotFoundError:
        recipes_data = pd.DataFrame()

    # Get unique values for Cuisine, Course, and Diet from the CSV file
    cuisine_options = [''] + [str(cuisine).strip("[]'") for cuisine in recipes_data['Cuisine'].unique().tolist()]
    course_options = [''] + [str(course).strip("[]'") for course in recipes_data['Course'].unique().tolist()]
    diet_options = [''] + [str(diet).strip("[]'") for diet in recipes_data['Diet'].unique().tolist()]

    # Display the dropdown menus with the unique values
    selected_cuisine = st.selectbox('Select Cuisine:', cuisine_options, format_func=lambda x: str(x))
    selected_course = st.selectbox('Select Course:', course_options, format_func=lambda x: str(x))
    selected_diet = st.selectbox('Select Diet:', diet_options, format_func=lambda x: str(x))

    # Filter the recipes based on the selected options
    filtered_recipes = recipes_data
    if selected_cuisine:
        filtered_recipes = filtered_recipes[filtered_recipes['Cuisine'].str.contains(selected_cuisine)]

    if selected_course:
        filtered_recipes = filtered_recipes[filtered_recipes['Course'].str.contains(selected_course)]

    if selected_diet:
        filtered_recipes = filtered_recipes[filtered_recipes['Diet'].str.contains(selected_diet)]

    if any([selected_cuisine, selected_course, selected_diet]) and not filtered_recipes.empty:
        st.write('Available recipes:')
        for recipe_name in filtered_recipes['RecipeName'].tolist():
            st.write(recipe_name)
    elif any([selected_cuisine, selected_course, selected_diet]) and filtered_recipes.empty:
        st.write('No recipes found.')

def Home():
    st.title("Welcome to Cookr - Your Ultimate Food Delivery App")
    st.image("Cookr_frontpage1.png", use_column_width=True)

    st.write("""
    ## Explore Delicious Cuisines and Dishes
    Discover a wide range of cuisines and dishes from local restaurants and chefs. Whether you're craving Indian, Continental, or something unique, Cookr has it all.

    ## How Cookr Works
    1. **Browse Menu:** Explore the diverse menu options from various restaurants.
    2. **Place Order:** Place your order with just a few clicks.
    3. **Track Delivery:** Track your order in real-time from the kitchen to your doorstep.

    ## Featured Restaurants
    Check out some of our featured restaurants offering mouth-watering dishes:

    - **Spice Haven:** Authentic Indian flavors
    - **Taste Buds Delight:** International cuisine at its best
    - **Veggie Delight Express:** A paradise for vegetarians

    ## Special Offers
    Enjoy exclusive discounts and special offers on selected dishes. Keep an eye out for daily deals and promotions.

    ## Contactless Delivery
    Your safety is our priority. Cookr ensures contactless delivery to provide you with a safe and convenient experience.

    ## Get Started
    Ready to embark on a culinary journey? Click the button below to start exploring Cookr!

    """)

    if st.button("Explore Cookr"):
        # Add code to navigate to the main app or explore menu options
        st.success("Let's start exploring! 🍔🍕🍣")

def main():
    
        
        st.sidebar.title("Cookr")
        
        page = st.sidebar.selectbox("Select a Page", ["Home Page", "Kitchen", "User",])
            
        if page ==  "Home Page":
                Home()
        elif page == "Kitchen":
                Add_Item()
        elif page == "User":
                Search()
                
        st.sidebar.write ("Developed by : ")
        st.sidebar.write("  Krithika")
        st.sidebar.write("  Rajaneha") 
      
        

if __name__ == "__main__":
    main()      
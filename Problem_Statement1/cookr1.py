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

def Add_Item():
    
    
    with open('model.pkl', 'rb') as model_file:
        model = joblib.load(model_file)
        
    with open('tfidf_vectorizer.pkl','rb') as model_file1:
        tfidf_vector = joblib.load(model_file1)
        
    # Take input from the user
    input_recipe = st.text_input("Enter the recipe name: ")
    if st.button('Add'):

        input_tfidf = tfidf_vector.transform([input_recipe])

        # Make predictions using the trained model
        predicted_labels = {}
        for label, classifier in model.items():
            predicted_labels[label] = classifier.predict(input_tfidf)

        st.write("Predicted Labels:")
        #print(predicted_labels)
        for label, prediction in predicted_labels.items():
            st.write(f"{label}: {prediction[0]}")
            
        data = {'RecipeName': [input_recipe],
            'Cuisine': [predicted_labels['Cuisine'][0]],
            'Course': [predicted_labels['Course'][0]],
            'Diet': [predicted_labels['Diet'][0]]}

            
        df = pd.DataFrame(data)
        file_exists = os.path.isfile('recipes_data.csv')

        # Write to CSV with or without header
        df.to_csv('recipes_data.csv', mode='a', header=not file_exists, index=False)
            
def Search():
    st.title("Search for Food")
    
    cuisine_options = ['Indian', 'Continental', 'South Indian Recipes', 'North Indian Recipes','Gujarati Recipes']
    course_options = ['Breakfast', 'South Indian Breakfast','Lunch', 'Dinner', 'Dessert','Appetizer','Snack']
    diet_options = ['Vegetarian', 'Non Vegeterian','High Protein Vegetarian','Eggetarian']

    # Display the dropdown menus with a default placeholder
    selected_cuisine = st.selectbox('Select Cuisine:', [''] + cuisine_options)
    selected_course = st.selectbox('Select Course:', [''] + course_options)
    selected_diet = st.selectbox('Select Diet:', [''] + diet_options)

        
    # Read the recipes_data.csv file
    try:
        recipes_data = pd.read_csv('recipes_data.csv')
    except FileNotFoundError:
        recipes_data = pd.DataFrame()

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
    st.image("Cookr_frontpage.png", use_column_width=True)

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
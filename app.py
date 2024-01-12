# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 18:11:34 2024

@author: mdref
"""

import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Define the names of the courses
course_names = [
    "Structured Programming",
    "Structured programming (Sessional)",
    "OOP",
    "OOP (Sessional)",
    "Discrete Math",
    "Calculus",
    "English Skill Development",
    "Data Structure",
    "Data Structure (Sessional)",
    "Algorithm",
    "Algorithm (Sessional)",
    "Applied Statistics",
    "CP Lab",
    "DBMS",
    "DBMS (Sessional)",
    "Theory of Computing",
    "Operating Systems",
    "Operating Systems (Sessional)",
    "Computer Architecture",
    "Artificial Intelligence",
    "Machine Learning",
    "Compiler Design",
    "Engineering Management",
    "Thesis",
    "Total Problems Solved"
]

# Define grade options for most courses
grade_options = [4.0, 3.75, 3.5, 3.25, 3.0, 2.75, 2.5, 2.25, 2]

# Define grade options for "Total Problems Solved"
problems_solved_grade_options = {
    2: "1-100 problems solved",
    2.5: "101-300 problems solved",
    3: "301-600 problems solved",
    3.5: "601-1000 problems solved",
    4: "1000+ problems solved"
}

# Load the trained machine learning models
log_reg_model = joblib.load('C:/Users/mdref/Downloads/TWP_Project/logistic_regression_model.pkl')
rf_model = joblib.load('C:/Users/mdref/Downloads/TWP_Project/random_forest_model.pkl')
gb_model = joblib.load('C:/Users/mdref/Downloads/TWP_Project/gradient_boosting_model.pkl')

# Define a function to preprocess user input and make predictions
def predict_strength(input_data, model):
    # Preprocess the input data (similar to what you did for training)
    scaler = StandardScaler()
    input_data_scaled = scaler.fit_transform(input_data)
    
    # Make predictions using the loaded model
    predictions = model.predict(input_data_scaled)
    return predictions

# Define the sections
sections = {
    "Home": """
    **Welcome to the CSE Student Strength Predictor: Unlocking Academic Potential**
    
    In the ever-evolving landscape of education, understanding a student's academic strength is paramount. The CSE Student Strength Predictor is a groundbreaking tool tailored for the world of Computer Science and Engineering (CSE) undergraduate students. This innovative project has a singular mission: to empower educators, students, and academic institutions with insights that unravel the intricate tapestry of academic performance and potential.""",
    
    "Predictor System": """
    **Predictor System: Precision Meets Potential**
    
    At the heart of our project lies the Predictor System, a culmination of cutting-edge machine learning models that breathe life into academic data. Imagine a world where a student's course grades, often seen as mere numbers, transform into meaningful insights. This system takes input in the form of grades for 25 carefully selected courses and, in return, offers a glimpse into the student's academic standing.
    
    The predictions generated are categorized into five distinct tiers:
    
    - **Average:** For those who have carved a moderate path in academia.
    - **Medium:** A stepping stone above average performance.
    - **Good:** Reflecting a strong academic foundation.
    - **Very Good:** Reserved for academic excellence.
    - **Excellent:** The pinnacle of achievement, reserved for the brightest stars.
    
    This system isn't a crystal ball; it's a scientific amalgamation of logistic regression, random forest, and gradient boosting models. The predictions aren't guesswork; they are reliable and precise indicators of academic prowess. Educators can tailor their approaches to individual needs, offer timely support to struggling students, and celebrate outstanding achievements. Meanwhile, students can chart their academic trajectory with confidence.""",
    
    "Motivation": """
    **Motivation: Illuminating the Path to Excellence**
    
    The genesis of this project lies in a profound motivation—to bridge the gap between data-driven insights and academic excellence. It's not just a tool; it's a beacon of guidance. We understand that every student is unique, with their own strengths and areas for improvement. With the CSE Student Strength Predictor, we aim to provide a nuanced understanding of their academic journey.""",
    
    "Collaborators": """
    **Collaborators: A Symphony of Expertise**

    The CSE Student Strength Predictor is the result of a harmonious collaboration between individuals passionate about education. They joined forces to bring this vision to life. Their combined expertise in data science, education, and technology ensures that the tool not only meets but exceeds expectations.

    **Project Contributors:**

    Name: Abu Md. Masbah Uddin  
    Github Profile: [Masbah Uddin](https://github.com/ma5bah)  
    Linkedin Account: [Masbah Uddin](https://www.linkedin.com/in/ma5bah/)  
    Portfolio: [Masbah Uddin](https://ma5bah.com/)


    Name: Tofayel Ahmmed Babu  
    Github Profile: [Tofayel Ahmmed](https://github.com/TofayelAhmmedBabu)  
    Linkedin Account: [Tofayel Ahmmed](https://www.linkedin.com/in/tofayelahmmedbabu/)  
    Portfolio: [Tofayel Ahmmed](https://tofayelahmmedbabu.vercel.app/)  

    Name: Md. Refaj Hossan  
    Github Profile: [Hossan R.](https://github.com/RJ-Hossan)   
    Linkedin Account: [Hossan R.](https://www.linkedin.com/in/mdrefajhossan/)  
    Portfolio: [Hossan R.](https://refaj-hossan.vercel.app/)  
    """,
    
    "Privacy": """
    **About: Upholding Privacy and Integrity**
    
    We understand the sensitivity of academic data. The CSE Student Strength Predictor operates with the utmost respect for privacy and data integrity. Our commitment to transparency ensures that users can trust the results and use them to enhance the academic experience.
    In conclusion, we invite you to embark on this academic journey with us. The CSE Student Strength Predictor is more than just a tool; it's a catalyst for academic excellence. Welcome to a world where data-driven insights illuminate the path to greatness."""
}

# Streamlit app header
st.title('CSE Student Strength Predictor')

# Navigation bar
st.sidebar.title('Navigation')
selected_section = st.sidebar.radio("Select your preference", list(sections.keys()))

# Displaying the selected section content
st.markdown(sections[selected_section])

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("[Github](https://github.com/RJ-Hossan)")
st.sidebar.markdown("[Mail 1](mailto:u1904001@student.cuet.ac.bd) | [Mail 2](mailto:u1904005@student.cuet.ac.bd) | [Mail 3](mailto:u1904007@student.cuet.ac.bd)")
st.sidebar.markdown("Developed by Team Arigato_CUET")  
st.sidebar.markdown("© CSE Student Strength Predictor, 2024 | Team Arigato_CUET")

# Input form for user to enter data in 2 columns
if selected_section == "Predictor System":
    st.header('Enter Course Grades')
    num_courses = len(course_names)
    num_columns = 2
    num_rows = num_courses // num_columns + (num_courses % num_columns > 0)
    user_input = []

    for i in range(num_rows):
        cols = st.columns(num_columns)
        for j in range(num_columns):
            idx = i * num_columns + j
            if idx < num_courses:
                course_name = course_names[idx]

                if course_name == "Total Problems Solved":
                    grade = cols[j].selectbox(f'{course_name}:', list(problems_solved_grade_options.keys()), format_func=lambda x: problems_solved_grade_options[x])
                else:
                    grade = cols[j].selectbox(f'Grade for {course_name}:', grade_options)

                user_input.append(grade)

    # Box to display the strength categories
    st.subheader('Strength Categories:')
    st.markdown("1 - Average")
    st.markdown("2 - Medium")
    st.markdown("3 - Good")
    st.markdown("4 - Very Good")
    st.markdown("5 - Excellent")

    if st.button('Predict'):
        # Convert user input into a DataFrame
        input_data = pd.DataFrame([user_input], columns=course_names)

        # Make predictions using the models
        log_reg_prediction = predict_strength(input_data, log_reg_model)
        rf_prediction = predict_strength(input_data, rf_model)
        gb_prediction = predict_strength(input_data, gb_model)

        # Display the predicted strength category for each model
        strength_categories = {1: 'Average', 2: 'Medium', 3: 'Good', 4: 'Very Good', 5: 'Excellent'}

        st.subheader('Predicted Strength Categories:')
        st.markdown(f'According to **Logistic Regression**: {strength_categories.get(int(log_reg_prediction[0]), "Unknown")}')
        st.markdown(f'According to **Random Forest**: {strength_categories.get(int(rf_prediction[0]), "Unknown")}')
        st.markdown(f'According to **Gradient Boosting**: {strength_categories.get(int(gb_prediction[0]), "Unknown")}')
